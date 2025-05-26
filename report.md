
## LongEval 2025 – Experiment Report

### 1. Objective

The aim of these experiments was to build and evaluate a scientific document retrieval pipeline for the LongEval 2025 challenge. We focused on hybrid approaches, combining classical (BM25) retrieval, dense retrieval models, and cross-encoder reranking.

---

### 2. Tested Approaches & Models

#### 2.1. Hybrid: BM25 + Dense Retrieval + Cross-Encoder

Our primary pipeline tested the following combination:

* **BM25 Index** (via Pyserini).
* **Dense retrieval** using SentenceTransformer-based models:

  * Models tested:

    * `malteos/scincl`
    * `BAAI/bge-base-en-v1.5`
    * `intfloat/e5-base-v2`
* **Reranking** with cross-encoders:

  * `cross-encoder/ms-marco-MiniLM-L-12-v2`
  * `BAAI/bge-reranker-base`

**Workflow:**

1. BM25 returns TOP-K candidates.
2. Dense model returns TOP-K candidates.
3. Candidates are merged, deduplicated, and reranked with a cross-encoder.

#### 2.2. Dense Retrieval + Cross-Encoder Only

For comparison, we also implemented a dense-only pipeline:

* Documents and queries encoded using dense models (e.g., `intfloat/e5-base-v2`).
* Top-K candidates selected by similarity.
* Final reranking with a cross-encoder.

---

### 3. Results & Observations

| Pipeline                                                                | Dense Model         | Cross-Encoder                         | nDCG\@10 |
| ----------------------------------------------------------------------- | ------------------- | ------------------------------------- | -------- |
| BM25 + malteos/scincl + cross-encoder/ms-marco-MiniLM-L-12-v2           | malteos/scincl      | cross-encoder/ms-marco-MiniLM-L-12-v2 | \~0.28   |
| BM25 + BGE-base-en-v1.5 + BGE-reranker-base                             | BGE-base-en-v1.5    | BGE-reranker-base                     | \~0.40   |
| Dense only: intfloat/e5-base-v2 + cross-encoder/ms-marco-MiniLM-L-12-v2 | intfloat/e5-base-v2 | cross-encoder/ms-marco-MiniLM-L-12-v2 | \~0.68   |

**Key findings:**

* The strictly hybrid pipeline (BM25 + dense + reranker) performed **worse** than dense retrieval with reranking.
* Switching from `scincl` to `bge-base-en-v1.5` gave a small improvement (nDCG\@10: \~0.28 → \~0.40).
* **Best results** were achieved with dense retrieval (`intfloat/e5-base-v2`) followed by cross-encoder reranking (nDCG\@10 ≈ 0.68).

**Takeaways:**

* Adding BM25 did **not** improve results—in fact, it hurt performance in this setup.
* The most effective pipeline was: **dense retrieval + cross-encoder reranking**.
* Preprocessing (concatenating title + abstract, lowercasing, prompt engineering) made some difference, but **model choice mattered most**.

---

### 4. Technical Issues

* **Docker & TIRA Submission:**

  * Locally, the pipeline ran smoothly, but there were issues when submitting via TIRA:

    * The `ir-metadata.yml` file had to be present in the repo and copied to the Docker image.
    * **No Internet on TIRA!** Models from HuggingFace needed to be pre-downloaded/cached; otherwise, the pipeline failed when loading models.
  * After several fixes, we were able to submit code and results, which are now visible on tira.io.

---

### 5. Summary

* **Best result:** nDCG\@10 ≈ 0.68 for dense retrieval (`intfloat/e5-base-v2`) + cross-encoder reranking.
* Hybrid pipelines with BM25 did **not** provide additional gains.
* Main challenges: TIRA’s offline environment and strict metadata requirements.
* **Final results and code were successfully submitted via tira-cli and are visible on tira.io.**
