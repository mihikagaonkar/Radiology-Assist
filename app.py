import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import pinecone
from typing import Optional, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
import torch

pc = None

def init_clients():
    global pc
    key = os.getenv("PINECONE_API_KEY", "")
    
    pc = pinecone.Pinecone(api_key=key)

def get_text_embedding(text: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    bert_model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = bert_model(**inp)
        emb = out.last_hidden_state[:,0,:]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32)

def get_image_embedding(image_bytes: bytes) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True)
    return emb.astype(np.float32)

def combine_embeddings(img_emb: Optional[np.ndarray], txt_emb: Optional[np.ndarray], w_img: float, w_txt: float) -> np.ndarray:
    if img_emb is None and txt_emb is None:
        raise ValueError("At least one modality is required")
    if img_emb is None:
        v = txt_emb / np.linalg.norm(txt_emb)
        return v
    if txt_emb is None:
        v = img_emb / np.linalg.norm(img_emb)
        return v
    st.info(f"Image emb norm before weighting: {img_emb.shape}  {np.linalg.norm(img_emb)}")
    v = np.concatenate([txt_emb, img_emb])
    if np.linalg.norm(v) == 0:
        return v
    return (v / np.linalg.norm(v)).astype(np.float32)
    # final_vec = np.concatenate([txt_emb, img_emb])
    # return final_vec.astype(np.float32)

import json
import streamlit as st
from typing import Tuple, Any, Dict, List

def _extract_dimension_from_desc(desc: Any) -> Optional[int]:
    if desc is None:
        return None
    if isinstance(desc, int):
        return desc
    if isinstance(desc, str):
        try:
            j = json.loads(desc)
            desc = j
        except Exception:
            pass
    if isinstance(desc, dict):
        candidates = []
        keys_to_check = [
            "dimension",
            "config",
            "index_config",
            "indexConfig",
            "metadata",
            "index_metadata",
            "index_metadata.config",
            "index_stats"
        ]
        for k in ("dimension",):
            v = desc.get(k) if isinstance(desc, dict) else None
            if isinstance(v, int):
                return int(v)
            if isinstance(v, str) and v.isdigit():
                return int(v)
        for outer in ("config", "index_config", "indexConfig", "metadata", "index_metadata", "index_stats"):
            sub = desc.get(outer)
            if isinstance(sub, dict):
                for k in ("dimension", "dim", "vector_size"):
                    vv = sub.get(k)
                    if isinstance(vv, int):
                        return int(vv)
                    if isinstance(vv, str) and vv.isdigit():
                        return int(vv)
        # last-ditch: scan all nested ints that look like a dimension
        def find_ints(d):
            out = []
            if isinstance(d, dict):
                for val in d.values():
                    out += find_ints(val)
            elif isinstance(d, list):
                for val in d:
                    out += find_ints(val)
            elif isinstance(d, int):
                out.append(d)
            return out
        ints = find_ints(desc)
        plausible = [i for i in set(ints) if 32 <= i <= 8192]
        if len(plausible) == 1:
            return plausible[0]
    return None


def get_index_dimension_safe(index_name: str) -> int:
    index = pc.Index('mimic-multimodal')
    desc = index.describe_index_stats()
    dimension = desc['dimension']
    return dimension


def adjust_vector_to_dim(vec: np.ndarray, target_dim: int, strategy: str = "auto") -> Tuple[np.ndarray, bool, str]:
    cur = int(vec.shape[0])
    if cur == target_dim:
        return vec, False, "none"
    if cur > target_dim:
        if strategy in ("truncate", "auto"):
            return vec[:target_dim].astype(np.float32), True, f"truncated {cur}->{target_dim}"
            raise RuntimeError(f"Vector length {cur} > target {target_dim} and truncation not allowed")
    else:
        if strategy in ("pad", "auto"):
            pad_len = target_dim - cur
            pad = np.zeros((pad_len,), dtype=vec.dtype)
            return np.concatenate([vec, pad]).astype(np.float32), True, f"padded {cur}->{target_dim}"
        raise RuntimeError(f"Vector length {cur} < target {target_dim} and padding not allowed")

def query_index(index_name: str, vector: np.ndarray, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None, auto_adjust: bool = True) -> Any:
    target_dim = get_index_dimension_safe(index_name)
    vec_dim = int(vector.shape[0])
    adjusted = False
    adjust_msg = ""
    if vec_dim != target_dim:
        if auto_adjust:
            vector, adjusted, adjust_msg = adjust_vector_to_dim(vector, target_dim, strategy="auto")
            st.warning(f"Adjusted query vector: {adjust_msg}. This is a temporary workaround — re-create index with the correct embed dim for best results.")
        else:
            raise RuntimeError(f"Vector dimension {vec_dim} does not match index dimension {target_dim}")
    try:
        index = pc.Index(index_name)
        if filter_meta:
            res = index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True, filter=filter_meta)
        else:
            res = index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
        return res
    except Exception as e:
        raise RuntimeError(f"Query to index '{index_name}' failed: {e}")
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def rerank_hits_with_report_embeddings(query_img_emb, query_txt_emb, hits, get_text_embedding_fn, alpha=0.5):
    # alpha weights image similarity vs text similarity (0.0..1.0)
    reranked = []
    for hit in hits:
        meta = hit.get("metadata", {})
        report = meta.get("report_text", "") or ""
        report_emb = get_text_embedding_fn(report) if report else None
        img_sim = cosine(query_img_emb, np.array(meta.get("img_emb")) ) if meta.get("img_emb") is not None else (cosine(query_img_emb, hit["vector"]) if "vector" in hit else 0.0)
        txt_sim = cosine(query_txt_emb, report_emb) if (query_txt_emb is not None and report_emb is not None) else 0.0
        combined = alpha * img_sim + (1 - alpha) * txt_sim
        reranked.append((combined, hit))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [h for score,h in reranked]

def format_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    meta = hit.get("metadata", {})
   
    return {
        "id": hit.get("id"),
        "score": hit.get("score"),
        "image_url": meta.get("image_url"),
        "report": meta.get("report")
    }

def render_results(results: Any):
    matches = results.get("matches", []) if isinstance(results, dict) else results["matches"]
    for m in matches:
       
        r = format_hit(m)
        # st.info(f"Raw match: {r}")
        st.write(f"**Match — {r['id']} (score {r['score']:.4f})**")
        cols = st.columns([1,2])
        with cols[0]:
            if r["image_url"]:
                try:
                    import requests
                    resp = requests.get(r["image_url"], timeout=5)
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    st.image(img, use_column_width=True)
                except Exception:
                    st.text("Image not available")
        with cols[1]:
            # st.markdown(f"**Diagnosis:** {r.get('diagnosis', '—')}")
            # st.markdown(f"**ICD:** {r.get('icd', '—')}")
            # st.markdown(f"**Agreement score:** {r.get('agreement', '—')}")
            # st.markdown(f"**Radiologist disagreement:** {r.get('radiologist_disagreement', '—')}")
            # if r.get('guideline_link'):
            #     st.markdown(f"**Guideline:** [{r['guideline_link']}]({r['guideline_link']})")
            # st.markdown("**Report**")
            st.text_area("Report", value=r.get("report", ""), height=180, key=f"report_{r['id']}")
            st.markdown("---")

def main():
    st.set_page_config(page_title="Radiology Guided RAG", layout="wide")
    st.title("Guided RAG — Radiology Assist")
    init_clients()
    index_name = st.sidebar.text_input("Pinecone index name", value=os.getenv("PINECONE_INDEX_NAME", "mimic-multimodal"))
    top_k = st.sidebar.slider("Top K", 1, 20, 5)
    w_img = st.sidebar.slider("Image weight", 0.0, 1.0, 0.6)
    w_txt = st.sidebar.slider("Text weight", 0.0, 1.0, 0.4)
    only_disagreements = st.sidebar.checkbox("Only cases with radiologist disagreement (uncertainty-focused)")
    show_debug = st.sidebar.checkbox("Show index debug info")
    st.sidebar.markdown("---")
    txt = st.text_area("Clinical findings / report text (optional)")
    uploaded_file = st.file_uploader("Upload CT/X-ray/MRI image (optional)", type=["png","jpg","jpeg","tiff","dcm"])
   
    if st.button("Retrieve matches"):
        img_emb = None
        txt_emb = None
        if uploaded_file is not None:
            content = uploaded_file.read()
        
            img_emb = get_image_embedding(content)
            st.info(f"Image embedding shape: {img_emb.shape}")
            if img_emb is None:
                st.info(f"Image embedding failed: {e}")
                return
        if txt and txt.strip():
            
            txt_emb = get_text_embedding(txt)
            st.info(f"Text embedding shape: {txt_emb.shape}")
            if txt_emb is None:
                st.info(f"Text embedding failed: {e}")
                return
        if img_emb is None and txt_emb is None:
            st.error("Please provide at least an image or text input")
            return
        vector = combine_embeddings(img_emb, txt_emb, w_img, w_txt)
        st.info(f"Combined embedding shape: {vector.shape}")
        filter_meta = {"radiologist_disagreement": {"$eq": True}} if only_disagreements else None
        try:
            if show_debug:
                try:
                    dims = get_index_dimension_safe(index_name)
                    print(dims)
                    st.info(f"Index '{index_name}' dimension: {dims}")
                    print(int(vector.shape[0]))
                    st.info(f"Query vector dimension before adjust: {int(vector.shape[0])}")
                except Exception as e:
                    st.warning(f"Could not fetch index info: {e}")
            res = query_index(index_name=index_name, vector=vector, top_k=top_k, filter_meta=filter_meta, auto_adjust=True)
        except Exception as e:
            st.error(f"Query failed: {e}")
            return
        render_results(res)

if __name__ == "__main__":
    main()
