
---

```markdown
# PennyBot_LLM_Agentic_RAG

**PennyBot reborn as an LLM‑Agentic RAG Chatbot**  
Dockerized, CUDA‑accelerated, TTFT tracked, hallucination taxonomy logged, and orchestrated end‑to‑end with sustainable low‑token, low‑energy retrieval.

---

## 📂 Repository Structure

AgenticRAG/
├── .dockerignore
├── .env
├── Dockerfile
├── requirements.txt
├── run_all.bat
├── run_log.txt
├── all_questions_tagged.csv
├── financebench_open_source.jsonl
├── build_index.py          # Vector store construction
├── chat_cli.py             # Command-line chatbot interface
├── etl.py                  # Extract-transform-load pipeline
├── evaluate.py             # Evaluation harness (EM, F1, TTFT, hallucination taxonomy)
├── generate_corpus.py      # Corpus generation scripts
├── ingest_and_filter.py    # Ingestion + filtering logic
├── pinecone_rest.py        # Pinecone API wrapper
├── rag_agent_library.py    # Core agent orchestration library
├── seed_from_jsonl.py      # Seed corpus from JSONL
├── __pycache__/            # Python cache (ignored in .gitignore)
└── .vscode/                # VS Code settings (ignored in .gitignore)

---

## 📘 Part I. Mathematical Foundations (Textbook Mode)

### 1. Document Representation
Let the dataset be:
\[
D = \{d_1, d_2, \dots, d_n\}
\]
Each document \(d_i\) is segmented into smaller textual chunks:
\[
C = \{c_{11}, c_{12}, \dots, c_{nm}\}
\]

### 2. Embedding Function
Each chunk \(c \in C\) is mapped into a high‑dimensional vector space via an embedding function \(f\):
\[
v_c = f(c) \in \mathbb{R}^d
\]

### 3. Vector Store Construction
All chunk embeddings are stored in a FAISS index:
\[
V = \{v_{c_1}, v_{c_2}, \dots, v_{c_k}\}
\]
Similarity between a query vector \(q\) and a chunk vector \(v_c\) is computed using cosine similarity:
\[
\text{sim}(q, v_c) = \frac{q \cdot v_c}{\|q\| \cdot \|v_c\|}
\]

### 4. Retrieval
Given a user query \(q\), we embed it:
\[
q = f(q)
\]
We then retrieve the top‑k most similar chunks:
\[
R(q) = \text{arg top‑k}_{c \in C} \ \text{sim}(q, v_c)
\]

### 5. Augmented Generation
The retrieved chunks \(R(q)\) are concatenated with the query and passed to the language model:
\[
\text{Answer}(q) = \text{LLM}(q \oplus R(q))
\]
Here, \(\oplus\) denotes concatenation of query and retrieved context.

### 6. Evaluation Metrics
- **Exact Match (EM)**: binary check if normalized prediction = gold.  
- **Token F1**: harmonic mean of precision/recall over token overlap.  
- **TTFT**: time to first token.  
- **Total Latency**: end‑to‑end wall‑clock time.  
- **Hallucination Taxonomy**: {grounded, unsupported_numeric, unsupported_claim}.  


## 📐 Prompt Engineering Math

### Weighted Context Fusion
P(q) = q ⊕ Σ αᵢ · cᵢ  
- q = query  
- cᵢ = retrieved chunk  
- αᵢ = weight coefficient (similarity, token budget, energy cost)

### Token + Energy Cost Function
Cost(R) = λ · Tokens(R) + μ · Energy(R)

### TTFT Metric
TTFT = t_first − t_request  
Latency = t_last − t_request

### Hallucination Taxonomy
H(x) =
- 0 → grounded in retrieved context  
- 1 → unsupported numeric claim  
- 2 → unsupported textual claim

### Constraint‑Driven Prompt
Prompt(q) = LLM(q ⊕ R(q) | Constraints)

---

## 🔑 API Keys

To run PennyBot_LLM_Agentic_RAG you’ll need free API keys:

- [Together AI](https://api.together.xyz/) → for cost‑efficient embeddings and hosted inference
- [Pinecone](https://www.pinecone.io/) → for scalable vector database
- (Optional) Hugging Face Hub → for dataset pulls and model hosting

Add them to your `.env` file:

TOGETHER_API_KEY=your_together_key  
PINECONE_API_KEY=your_pinecone_key

---
Cost-Benefit Analysis

Yes — if this README is going to be a **saga**, it needs both the *practical links* (where to grab free API keys) and the *numerical testimony* (your end‑to‑end cost slicing). Right now it reads like a textbook, but you want it to feel like a fellowship epic: math, code, lore, and economics all braided together.

---

## 🔑 API Keys Section (README.md)

```markdown
## 🔑 API Keys

To run PennyBot_LLM_Agentic_RAG you’ll need free API keys:

- [OpenAI](https://platform.openai.com/) → for GPT models and embeddings
- [Together AI](https://api.together.xyz/) → for cost‑efficient embeddings and hosted inference
- [Pinecone](https://www.pinecone.io/) → for scalable vector database
- (Optional) Hugging Face Hub → for dataset pulls and model hosting

Add them to your `.env` file:

OPENAI_API_KEY=your_openai_key  
TOGETHER_API_KEY=your_together_key  
PINECONE_API_KEY=your_pinecone_key


---


## 💸 End‑to‑End Cost Optimization

You engineered this like an economist. Here’s how to show it off:

### 1. Token Cost Function
\[
\text{Cost}_{\text{tokens}} = \lambda \cdot \text{InputTokens} + \mu \cdot \text{OutputTokens}
\]

### 2. Retrieval Cost Function
\[
\text{Cost}_{\text{retrieval}} = \alpha \cdot k + \beta \cdot \text{Latency}
\]

### 3. Total Pipeline Cost
\[
\text{Cost}_{\text{total}} = \text{Cost}_{\text{tokens}} + \text{Cost}_{\text{retrieval}} + \text{Energy}_{\text{CUDA}}
\]

---

## 📊 Example Numbers (Your Saga)

- **OpenAI embeddings**: ~$0.10 per 1K queries (high‑fidelity, but pricier).  
- **Together embeddings**: ~$0.02 per 1K queries (optimized, fellowship‑grade).  
- **Pinecone storage**: ~$0.25 per GB/month (scales with corpus size).  
- **CUDA acceleration**: negligible marginal cost once GPU is provisioned.  
- **End‑to‑end pipeline**: you benchmarked ~84.5% accuracy with **100% coverage** at **< $0.05/query**.

---


## 📑 Part II. Codebook Translation (Developer Manual)

### Environment Setup
```bash
pip install langchain==0.3.7 langchain-community==0.3.7 \
            langchain-openai==0.3.7 langchain-together==0.3.7 \
            faiss-cpu python-dotenv pandas datasets scikit-learn tqdm PyYAML


### .env File
```env
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
EMBEDDING_PROVIDER=openai


### Retrieval + Generation
```python
retriever = get_retriever(index_path)
docs = retriever.retrieve(query, top_k=5)
chunks = [d.page_content for d in docs]
gen_resp = call_rag_generator(query, chunks)


### Evaluation Harness
- Logs EM, F1, hallucination type, complexity flag.  
- Tracks TTFT, total latency, input/output tokens.  
- Appends results to `results_tagged.csv`.  
- Prints summary averages for fellowship‑grade reproducibility.

---

## ✅ Summary
PennyBot’s resurrection is not just a chatbot. It is:
- A **CUDA‑powered, Docker‑hardened RAG pipeline**.  
- A **mathematical textbook** (Part I) and **developer codebook** (Part II).  
- A **fellowship artifact**: every eval request stamped with time, tokens, hallucination taxonomy, and reproducibility.



---

# 📚 References

## Core Frameworks
- **VeritasFi (2025)** — Hybrid retrieval + reranking for financial QA.  
  *Informed PennyBot’s hybrid retriever design (CAKC, reranker practices).*

- **Multi‑HyDE (2025)** — Hypothetical document embeddings.  
  *Inspired multi‑hop reasoning, query diversification, and recall curve tracking.*

- **FinSage (2025)** — Multi‑modal retrieval, hallucination reduction.  
  *Guided hallucination taxonomy, DPO reranker, and compliance‑critical retrieval.*

- **Financial Report Chunking for Effective RAG (2024)** — Element‑based chunking.  
  *Anchored PennyBot’s element‑aware chunking and metadata logging.*

- **FinQANet (2022)** — Program‑of‑thought reasoning.  
  *Influenced step‑by‑step reasoning and interpretable outputs.*

---

## Baselines
- **LightRAG (2022)** — Dense retrieval baseline.  
- **GraphRAG (2022)** — Graph‑structured retrieval baseline.  
- **BM25 (2009)** — Sparse retrieval baseline.  
- **FAISS (2017)** — Dense retrieval baseline.  
- **BM25 + FAISS (2019)** — Hybrid sparse‑dense baseline.  

*These baselines contextualize PennyBot’s resurrection: moving beyond dense/sparse hybrids into agentic orchestration.*

---

## Statistical Methods
- **Efron & Tibshirani (1993)** — Bootstrap confidence intervals.  
  *Used for reproducible paired comparisons.*  

- **Wilcoxon (1945)** — Signed‑rank test.  
  *Applied for nonparametric paired EM/F1 comparisons.*  
