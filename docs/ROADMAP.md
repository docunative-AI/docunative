# Roadmap: Edge Document Intelligence for Global Newcomers

## Project at a Glance

**What we’re building:** A privacy-first, fully on-device system that lets users ask questions about foreign-language legal documents in their own native language, with answers grounded in the source text.
**Core model:** Tiny Aya (3.35B, 70 languages) - Global + Regional specialists (Earth / Fire / Water).
**Key technology:** PyMuPDF + ChromaDB + BGE-M3 + `llama.cpp` + mDeBERTa.
**Privacy guarantee:** 100% on-device. No document ever leaves the user’s machine.
**Phase 2 Deadline:** March 24 (Working Demo + Preliminary Results).

---

## The System Architecture (Phase 2)

```text
USER ("What is my deposit amount?")
  │
  ▼
[ Gradio UI ]
  │
  ▼
[ PDF Extraction ] -> PyMuPDF splits document into 400-token chunks (80 overlap)
  │
  ▼
[ Embeddings ] -> BAAI/bge-m3 converts chunks & query into cross-lingual vectors
  │
  ▼
[ Vector Store ] -> ChromaDB (Local) retrieves Top-3 closest chunks
  │
  ▼
[ Generation ] -> Raw `llama-server` running Tiny Aya GGUF via HTTP requests.
  │               (Outputs using strict Markdown delimiters)
  ▼
[ Validation ] -> Regex + Pydantic ensures clean {answer, source_quote} JSON
  │
  ▼
[ NLI Check ] -> mDeBERTa-v3-base-xnli checks for Hallucinations (Entail/Contradict)
  │
  ▼
USER (Sees Answer + Highlighted Source Clause + Hallucination Badge)
```
