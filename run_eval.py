# run_eval.py
import re
import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

from rag_agent_library import (
    get_pinecone_vectorstore,
    create_rag_pipeline,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    PINECONE_INDEX_NAME
)
from langchain_together import ChatTogether, TogetherEmbeddings

# ---------------------------
# Initialization
# ---------------------------
def initialize_pipeline():
    llm = ChatTogether(model=LLM_MODEL_NAME, temperature=0.1)
    embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = get_pinecone_vectorstore(embeddings, force_reseed=False)
    if vectorstore is None:
        raise ConnectionError("Vector store failed to initialize or connect. Please seed the index.")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    pipeline = create_rag_pipeline(retriever=retriever, llm=llm)
    return pipeline, retriever

PIPELINE = None
RETRIEVER = None
try:
    PIPELINE, RETRIEVER = initialize_pipeline()
    print("✅ RAG Pipeline Initialized successfully.")
except Exception as e:
    print(f"CRITICAL SETUP FAILURE: {e}")

# ---------------------------
# Runner
# ---------------------------
def run_rag_generator(query: str) -> Dict[str, Any]:
    if PIPELINE is None:
        return {"answer": "ERROR: Pipeline not initialized.", "retrieved_docs": [], "input_tokens": 0, "output_tokens": 0}
    start_first = time.time()
    resp = PIPELINE.invoke({"question": query, "chat_history": []})
    ttft = time.time() - start_first
    # Ensure fields exist
    answer = resp["answer"] if isinstance(resp, dict) and "answer" in resp else str(resp)
    docs = resp.get("retrieved_docs", [])
    return {
        "answer": answer,
        "retrieved_docs": docs,
        "ttft": ttft,
        # placeholders; replace with real counts if you add a tokenizer/callback
        "input_tokens": resp.get("input_tokens", 0),
        "output_tokens": resp.get("output_tokens", 0),
    }

# ---------------------------
# Metrics
# ---------------------------
def normalize(text):
    return re.sub(r'\W+', ' ', (text or "").lower()).strip()

def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)

def token_f1(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)

def is_multi_hop_or_temporal(question):
    keywords = ["compare", "difference", "trend", "change", "growth", "decline", "between", "across", "over time", "historical"]
    return any(kw in question.lower() for kw in keywords)

def classify_hallucination(pred, gold, chunks):
    joined = " ".join(chunks)
    if normalize(gold) and normalize(gold) in normalize(joined):
        return "grounded"
    if re.search(r'\d', pred or "") and not any(re.search(r'\d', c or "") for c in chunks):
        return "unsupported_numeric"
    return "unsupported_claim"

# ---------------------------
# Eval loop
# ---------------------------

def evaluate_model(retriever, test_csv, out_csv, max_samples=1000, top_k=5):
    if PIPELINE is None or retriever is None:
        print("Evaluation SKIPPED due to critical setup failure.")
        return pd.DataFrame()

    # --- DATA LOADING CHANGE STARTS HERE ---
    test_path = Path(test_csv)
    if test_path.suffix == '.csv':
        df = pd.read_csv(test_path)
    elif test_path.suffix == '.jsonl':
        data = []
        try:
            with open(test_path, 'r') as f:
                for line in f:
                    # JSONL is one JSON object per line
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Error reading JSONL file: {e}")
            return pd.DataFrame()
    else:
        raise ValueError(f"Unsupported file format: {test_path.suffix}. Use .csv or .jsonl")
    # --- DATA LOADING CHANGE ENDS HERE ---

    results = []

    for i, row in df.iterrows():
        if i >= max_samples:
            break

        q = row["question"]
        gold = row.get("answer", "")

       # --- START OF TIMERS ---
        
        # Generation timing (this now includes retrieval)
        start = time.time()
        # This is now the ONLY call. It gets the answer AND the docs.
        gen = run_rag_generator(q) 
        end = time.time()

        # 2. GET DOCS AND ANSWER FROM THE 'gen' DICTIONARY
        pred = gen.get("answer", "").strip()
        docs = gen.get("retrieved_docs", []) # <-- GET DOCS FROM HERE

        # 3. PROCESS THE DOCS (same as before)
        chunks = [getattr(d, "page_content", "") for d in docs]
        chunk_meta = [getattr(d, "metadata", {}) for d in docs]
        
        # --- END OF TIMERS ---
        
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        halluc_type = classify_hallucination(pred, gold, [normalize(c) for c in chunks])
        is_complex = is_multi_hop_or_temporal(q)

        total_time = end - start
        ttft = gen.get("ttft", None)
        input_tokens = gen.get("input_tokens", None)
        output_tokens = gen.get("output_tokens", None)

        results.append({
            "id": row.get("id", i),
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": int(em),
            "f1": f1,
            "hallucination": halluc_type,
            "is_complex": is_complex,
            "retrieved_chunks": json.dumps(
                [{"text": c[:300], "meta": m} for c, m in zip(chunks, chunk_meta)]
            ),
            "timestamp": time.time(),
            "ttft": ttft,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })

        print(f"{i}: EM={int(em)} F1={f1:.2f} Halluc={halluc_type} Complex={is_complex} TTFT={ttft if ttft is not None else 0:.2f}s Total={total_time:.2f}s")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    dfres = pd.DataFrame(results)
    dfres.to_csv(out_csv, index=False)

    print("--- Summary ---")
    print("Exact Match:", dfres["em"].mean() if len(dfres) else 0)
    print("F1:", dfres["f1"].mean() if len(dfres) else 0.0)
    print("Hallucination rates:", dfres["hallucination"].value_counts(normalize=True).to_dict() if len(dfres) else {})
    print("Complex EM:", dfres[dfres["is_complex"]]["em"].mean() if len(dfres) and dfres["is_complex"].any() else "N/A")
    print("Avg TTFT:", dfres["ttft"].mean() if len(dfres) else 0.0)
    print("Avg Total Time:", dfres["total_time"].mean() if len(dfres) else 0.0)
    return dfres

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to the test dataset (CSV or JSONL).")
    parser.add_argument("--out", required=True, help="Output path for the results CSV.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate.")
    parser.add_argument("--top_k", type=int, default=5, help="Top K documents to retrieve.")
    args = parser.parse_args()

    evaluate_model(RETRIEVER, args.test, args.out, max_samples=args.max_samples, top_k=args.top_k)