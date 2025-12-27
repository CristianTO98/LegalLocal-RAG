# ⚖️ LegalLocal RAG

> **Air-Gapped Legal Research Assistant** - 100% Offline RAG System for U.S. Attorneys

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Privacy: Air-Gapped](https://img.shields.io/badge/Privacy-Air--Gapped-green.svg)]()

## 🎯 Overview

LegalLocal RAG is a **privacy-first** legal document analysis tool designed for attorneys who need to process sensitive documents without any risk of data exposure. The system runs **100% offline** on standard laptop hardware (Ryzen 5, 16GB RAM, no GPU required).

### Key Features

- 🔒 **True Air-Gap Compliance**: Zero network calls at runtime
- 🧠 **Dual Model Support**: Speed (3B) vs Precision (8B) modes
- 📄 **PDF Analysis**: Upload and analyze legal documents
- 🔍 **RAG Architecture**: Retrieval-Augmented Generation for accurate responses
- 👁️ **Visual Verification**: See the actual PDF pages where information was found
- 🛡️ **Anti-Hallucination Auditor**: Secondary validation of AI responses
- 💻 **CPU-Optimized**: Runs on standard hardware without GPU

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LegalLocal RAG                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │  Streamlit  │───▶│  LangChain   │───▶│  llama-cpp-python       │ │
│  │     UI      │    │ Orchestrator │    │  (Embedded LLM Engine)  │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│         │                  │                        │                │
│         ▼                  ▼                        ▼                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │   PyMuPDF   │    │   ChromaDB   │    │  HuggingFace Embeddings │ │
│  │ (PDF Render)│    │(Vector Store)│    │  (all-MiniLM-L6-v2)     │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    LOCAL MODELS (No API Calls)                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf  (Speed Mode)     │  │
│  │  /models/Llama-3.1-8B-Instruct-Q4_K_M.gguf  (Precision Mode) │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores (Ryzen 5/Intel i5) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| GPU | Not required | Not required |

### Software Requirements

- Python 3.10 or higher
- Windows 10/11, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LegalLocal-RAG.git
cd LegalLocal-RAG
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

Download GGUF models and place them in the `/models` directory:

#### Speed Mode (Llama 3.2 3B) - ~2 GB
```bash
# Using huggingface-cli
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models

# Or download manually from:
# https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
```

#### Precision Mode (Llama 3.1 8B) - ~4.5 GB
```bash
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models

# Rename to match expected filename:
mv ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf ./models/Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Select Model Mode

In the sidebar, choose between:
- **Speed Mode (3B)**: Faster responses, best for quick queries
- **Precision Mode (8B)**: Higher accuracy, best for complex legal analysis

### Step 2: Upload Document

Upload a PDF document using the sidebar uploader. The system will:
1. Extract text from all pages
2. Split into semantic chunks (800 chars, 150 overlap)
3. Generate embeddings locally
4. Create an in-memory vector index

### Step 3: Ask Questions

Enter your legal research question in the main area. Examples:
- "What are the termination clauses in this contract?"
- "What is the governing law provision?"
- "Summarize the liability limitations"

### Step 4: Review Response

The system displays:
1. **AI Response**: The generated answer with citations
2. **Audit Status**: PASS/FAIL indicator from the anti-hallucination module
3. **Evidence Panel**: Source text chunks and PDF page images for verification

## 🛡️ Security & Privacy

### Air-Gap Compliance

This application is designed for environments requiring complete data isolation:

- ✅ **No API Calls**: All inference runs locally via llama-cpp-python
- ✅ **No Telemetry**: No usage data collection or phone-home features
- ✅ **No External Dependencies at Runtime**: All models loaded from local disk
- ✅ **Memory-Only Processing**: Optional (default) in-memory vector storage
- ✅ **Source Verifiable**: All dependencies are open-source

### Suitable For

- HIPAA-regulated healthcare documents
- Attorney-client privileged communications
- ITAR/EAR controlled technical data
- Financial documents under SOX compliance
- Classified or sensitive government documents

## ⚙️ Configuration

### CPU Optimization Parameters

Located in `app.py`, the `load_llm()` function contains optimized parameters:

```python
n_threads=4       # Leave cores for OS (adjust based on your CPU)
n_batch=256       # Optimized for L3 cache
n_ctx=2048        # Context window (tokens)
temperature=0     # Deterministic output (critical for legal)
n_gpu_layers=0    # Force CPU-only execution
```

### Chunking Strategy

Optimized for legal documents:

```python
chunk_size=800    # Captures full legal clauses
chunk_overlap=150 # Preserves cross-references
```

## 🗂️ Project Structure

```
LegalLocal-RAG/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── models/            # GGUF model files
│   └── .gitkeep
├── chroma_db/         # Vector store persistence (optional)
│   └── .gitkeep
└── .cache/            # HuggingFace embeddings cache
    └── embeddings/
```

## 🔧 Troubleshooting

### "Model file not found"

Ensure model files are placed in `/models` with exact filenames:
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `Llama-3.1-8B-Instruct-Q4_K_M.gguf`

### "Out of memory"

- Close other applications
- Use Speed Mode (3B) instead of Precision Mode
- Reduce `n_ctx` to 1024 in `app.py`

### Slow First Response

First query after model load includes:
- Model weight loading (~10-30 seconds)
- Embedding model initialization (~5-10 seconds)
- Vector index creation (~5-15 seconds per 100 pages)

Subsequent queries are much faster.

### PDF Processing Errors

- Ensure PDF is not password-protected
- Check PDF is not corrupted
- Scanned PDFs require OCR (not currently implemented)

## 🔮 Future Enhancements

- [ ] OCR support for scanned documents
- [ ] Multi-document analysis
- [ ] Citation graph visualization
- [ ] Export to legal brief format
- [ ] Batch processing mode
- [ ] PyInstaller packaging for single-executable distribution

## 📄 License

MIT License - See LICENSE file for details.

## ⚠️ Disclaimer

This tool is for **research assistance only**. It does not constitute legal advice. Always verify AI-generated responses against original source documents. The developers assume no liability for decisions made based on this tool's output.

---

**Built for attorneys who take privacy seriously.** ⚖️🔒
