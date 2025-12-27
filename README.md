# ⚖️ LegalLocal RAG

> **Air-Gapped Legal Research Assistant** - 100% Offline RAG System for Legal Professionals

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Privacy: Air-Gapped](https://img.shields.io/badge/Privacy-Air--Gapped-green.svg)]()

---

## 🧠 Por Qué Este Sistema Es Diferente: La Filosofía Detrás del Diseño

### El Problema del RAG Tradicional en Documentos Legales

Cuando un abogado pregunta *"¿Cuál es la multa por retraso en la entrega?"*, un sistema RAG tradicional cortaría el documento en trozos de tamaño fijo (por ejemplo, 500 caracteres). El problema es que ese corte puede partir una cláusula a la mitad.

**Imagina este escenario:** La respuesta relevante está en la Cláusula 8.3 que dice:

> *"En caso de fuerza mayor debidamente acreditada, el contratista quedará eximido de cualquier penalización. Sin embargo, en caso de retraso injustificado, se aplicará una penalización del 5% diario sobre el valor del contrato..."*

Con chunking tradicional, la frase *"se aplicará una penalización del 5% diario"* podría estar en un chunk, mientras que la excepción de *"fuerza mayor"* quedó en el chunk anterior. El LLM te daría un consejo legal **erróneo** porque nunca vio la excepción.

### La Solución: Parent-Child Indexing con Structure-Aware Chunking

Este MVP implementa una estrategia de **indexación jerárquica** diseñada específicamente para documentos legales:

1. **Parents (Nodos Padre)**: Dividimos el documento respetando su estructura semántica natural. Los documentos legales tienen patrones claros: *"Artículo X"*, *"Cláusula Y"*, *"1.1."*, *"1.2."*. Cada Parent es una unidad semántica completa (una cláusula, un artículo).

2. **Children (Nodos Hijo)**: Cada Parent se subdivide en trozos pequeños (~256 tokens) que son los que realmente se indexan y buscan.

3. **El Truco**: Cuando buscas, el sistema encuentra un Child muy específico (alta precisión en la búsqueda). Pero cuando recupera el contexto para el LLM, **sube al Parent completo**. Así el modelo siempre tiene la cláusula entera con todas sus excepciones y matices.

### ¿Por Qué el Modelo BGE-small para Embeddings?

Elegimos **BAAI/bge-small-en-v1.5** por razones muy específicas:

| Característica | Valor | Por Qué Importa |
|----------------|-------|-----------------|
| **Tamaño** | ~130 MB | Cabe en memoria sin problema, carga instantánea |
| **Rendimiento** | State-of-the-Art | Supera a modelos más grandes en benchmarks de recuperación (MTEB) |
| **Contexto** | 512 tokens | Perfecto para los Child chunks de 256 tokens |
| **Velocidad CPU** | Optimizado | Latencia imperceptible en laptops de oficina |

> 💡 **Nota técnica**: BGE requiere un prefijo especial para queries: *"Represent this sentence for searching relevant passages: "* — esto ya está implementado en el sistema.

### ¿Por Qué Qwen 3 4B con Cuantización Q4?

Después de probar múltiples modelos (Qwen 3 4B thinking, Qwen 2.5 3B, Gemma 3n e4b Q4, Gemma 3 4B, Gemma 3n e4b Q8), **Qwen 3 4B Instruct** con cuantización Q4_K_M demostró ser el mejor balance entre:

- **Velocidad**: Mayor cantidad de tokens/segundo en CPU puro.
- **Inteligencia**: Respuestas coherentes y bien estructuradas para tareas legales, con capacidades superiores de razonamiento.
- **Consumo**: ~2.5 GB de RAM, ideal para portátiles de oficina.

Este sistema está **diseñado para correr en cualquier portátil de oficina sin GPU**. No necesitas una máquina gaming ni una workstation con CUDA. Un Ryzen 5 o Intel i5 con 16GB de RAM es más que suficiente.

---

## 🎯 Overview

LegalLocal RAG is a **privacy-first** legal document analysis tool designed for professionals who need to process sensitive documents without any risk of data exposure. The system runs **100% offline** on standard laptop hardware (Ryzen 5/Intel i5, 16GB RAM, no GPU required).

### Key Features

- 🔒 **True Air-Gap Compliance**: Zero network calls at runtime
- 🧠 **Qwen 3 4B**: State-of-the-art intelligence for CPU-based RAG
- 📄 **Smart PDF Analysis**: Structure-aware extraction with PyMuPDF
- 🔍 **Parent-Child RAG**: Hierarchical indexing for legal document precision
- 💬 **Modern Chat Interface**: Powered by Chainlit for a professional UX
- 📑 **Side-by-Side Citations**: Click on references to see original context immediately
- 💻 **CPU-Optimized**: Runs on standard office hardware without GPU

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LegalLocal RAG v2.0                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │  Chainlit   │───▶│  LlamaIndex  │───▶│  llama-cpp-python       │ │
│  │     UI      │    │ Orchestrator │    │  (Embedded LLM Engine)  │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│         │                  │                        │                │
│         ▼                  ▼                        ▼                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │   PyMuPDF   │    │   ChromaDB   │    │   BAAI/bge-small-en     │ │
│  │ (Block Ext.)│    │(Vector Store)│    │   (130MB Embeddings)    │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│              PARENT-CHILD HIERARCHICAL INDEXING                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Parent Chunks (1024-2048 tokens) - Full clauses/articles    │  │
│  │       ↓                                                       │  │
│  │  Child Chunks (256-512 tokens) - What gets indexed & searched│  │
│  │       ↓                                                       │  │
│  │  On retrieval: Child matches → Return Parent for context     │  │
│  └───────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    LOCAL MODEL (No API Calls)                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (~2.3 GB)       │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores (Ryzen 5/Intel i5) |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 10 GB free |
| GPU | **Not required** | **Not required** |

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

### 4. Download the Model

Download the Qwen 3 4B GGUF model and place it in the `/models` directory:

```bash
# Example using huggingface-cli for Qwen 2.5 (as Qwen 3 is a custom file)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  --include "Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

> **Note**: The model file should be named `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` in the models folder.

### 5. Run the Application

```bash
chainlit run app.py -w
```

The application will open in your default browser at `http://localhost:8000`

## 📖 Usage Guide

### Step 1: Upload Document

Upload a PDF document using the sidebar uploader. The system will:
1. Extract text by blocks using PyMuPDF (faster, cleaner extraction)
2. Create Parent chunks respecting document structure (articles, clauses)
3. Split Parents into Child chunks for precise indexing
4. Generate BGE embeddings for all Children
5. Create hierarchical vector index

### Step 2: Ask Questions

Enter your legal research question in the main area. Examples:
- "What are the termination clauses in this contract?"
- "What is the governing law provision?"
- "Summarize the liability limitations"

### Step 3: Review Response

The system displays:
1. **AI Response**: The generated answer with citations
2. **Evidence Panel**: Source text chunks and PDF page images for verification

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

Located in `app.py`, the LLM configuration contains optimized parameters:

```python
n_threads=4       # Leave cores for OS (adjust based on your CPU)
n_batch=256       # Optimized for L3 cache
n_ctx=2048        # Context window (tokens)
temperature=0     # Deterministic output (critical for legal)
n_gpu_layers=0    # Force CPU-only execution
```

### Chunking Strategy (Parent-Child)

Optimized for legal documents:

```python
# Parent Chunks - Full semantic units
parent_chunk_size=1536   # ~1024-2048 tokens, captures full clauses

# Child Chunks - What gets indexed
child_chunk_size=384     # ~256 tokens, precise search
child_chunk_overlap=64   # Smooth transitions
```

## 🗂️ Project Structure

```
LegalLocal-RAG/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── models/            # GGUF model files
│   └── Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
├── chroma_db/         # Vector store persistence (optional)
│   └── .gitkeep
└── .cache/            # Embeddings cache
    └── embeddings/
```

## 🔧 Troubleshooting

### "Model file not found"

Ensure model file is placed in `/models` with exact filename:
- `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`

### "Out of memory"

- Close other applications
- Reduce `n_ctx` to 1024 in `app.py`
- Ministral 3B is already optimized for low memory (~2GB)

### Slow First Response

First query after model load includes:
- Model weight loading (~10-20 seconds)
- BGE embedding model initialization (~3-5 seconds)
- Vector index creation (~5-10 seconds per 100 pages)

Subsequent queries are much faster.

### PDF Processing Errors

- Ensure PDF is not password-protected
- Check PDF is not corrupted
- Scanned PDFs require OCR (not currently implemented)

## 🔮 Future Enhancements

- [ ] OCR support for scanned documents
- [ ] Multi-document analysis
- [ ] Regex-based structure detection for different legal formats
- [ ] Export to legal brief format
- [ ] Batch processing mode

## 📄 License

MIT License - See LICENSE file for details.

## ⚠️ Disclaimer

This tool is for **research assistance only**. It does not constitute legal advice. Always verify AI-generated responses against original source documents. The developers assume no liability for decisions made based on this tool's output.

---

**Built for legal professionals who take privacy seriously.** ⚖️🔒
