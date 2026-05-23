# HalluciDetect

**Production-grade hallucination detection for LLM outputs.**

Most AI systems hallucinate. Most teams don't catch it until it's too late. HalluciDetect is a dual-model verification pipeline that checks every claim in an LLM response against a source document — with explanations, confidence scores, and auto-corrections.

Built from scratch. No wrapper libraries. No black boxes.

🔗 **Live Demo:** [hallucidetect.streamlit.app](https://hallucination-detector-3pw6466ruxwpgklahaqqfa.streamlit.app/) &nbsp;|&nbsp; ⭐ **Star if useful**

---

## The Problem

LLMs confidently hallucinate. A single wrong date, misattributed quote, or fabricated statistic can break trust in an entire system. Existing tools either:

- Use a single model with no cross-validation
- Rely on black-box APIs with no transparency
- Can't handle real-world document sizes (10+ pages)
- Give a verdict with no explanation

HalluciDetect solves all four.

---

## How it works (overview)

```
Source Document + LLM Response
           ↓
    Smart Chunking Engine
           ↓
  Hybrid Retrieval System
           ↓
    ┌──────────────┐
    │  NLI Model   │  ← statistical signal
    └──────┬───────┘
           │
    Smart API Filter
           │
    ┌──────▼───────┐
    │  LLM Judge   │  ← reasoning signal
    └──────┬───────┘
           │
    Weighted Fusion
           │
    Verdict + Explanation + Correction
```

The core insight: **one model is never enough.** A fast NLI model catches obvious contradictions. A large reasoning model explains subtle ones. A smart filter decides when to use which — keeping costs at zero.

---

## Key Features

| | |
|---|---|
| **PDF Support** | Real documents — 50+ pages, not just text snippets |
| **Hybrid Retrieval** | Two-signal search that never misses dates, codes, or proper nouns |
| **Dual-Model Pipeline** | Statistical NLI + Chain-of-Thought reasoning, cross-validated |
| **Smart API Filter** | Expensive model only called when needed — API quota preserved |
| **Persistent Storage** | Documents cached across sessions — no re-uploading |
| **Density Map** | Section-wise hallucination heatmap across the full document |
| **Auto-Correction** | Hallucinated claims rewritten using source document |
| **Audit Export** | Full verification report downloadable as structured text |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF |
| Vector Storage | ChromaDB |
| NLI Inference | DeBERTa-v3-large (GPU) |
| LLM Reasoning | Llama 3.3 70B via Groq |
| Retrieval | Hybrid semantic + keyword search |
| Embeddings | SentenceTransformers (local) |

---

## Local Setup

**1. Clone**
```bash
git clone https://github.com/mjshirsale/hallucination-detector
cd hallucination-detector
```

**2. Environment**
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**3. API Key**

Create `.env` in root:
```
GROQ_API_KEY=your_key_here
```

Free key at [console.groq.com](https://console.groq.com) — no credit card.

**4. Run**
```bash
streamlit run app.py
```

---

## What makes this different

**Most hallucination detectors are wrappers.** They call an API, get a score, and show it. HalluciDetect is a pipeline — each layer has a specific job, and the layers are designed to compensate for each other's weaknesses.

The NLI model is fast but doesn't explain. The LLM explains but is slow. The filter between them is what makes the system practical — it decides in real time which model to invoke, based on confidence thresholds. This is the design decision that separates a demo from a production tool.

The retrieval layer uses two fundamentally different search strategies simultaneously. Vector search finds semantically similar content. Keyword search finds exact matches. Neither alone is sufficient for real documents. The fusion of both is what makes 50-page PDFs work reliably.

Implementation details are intentionally not documented here.

---

## Roadmap

- [ ] Async pipeline for concurrent claim processing
- [ ] REST API for integration into existing RAG systems
- [ ] Domain-specific NLI fine-tuning
- [ ] Cloud-native vector storage for enterprise scale
- [ ] Structured JSON output for downstream automation

---

## Author

**Mohit J Shirsale**

[LinkedIn](https://www.linkedin.com/in/mohit-shirsale-7b900a31b/) &nbsp;·&nbsp; [GitHub](https://github.com/mjshirsale) &nbsp;·&nbsp; [Live Demo](https://hallucidetect.streamlit.app)

---

## License

MIT — use it, but build something original.