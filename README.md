# LegalLocal RAG: Total Privacy for Legal Document Analysis

This project was born out of a clear necessity: the legal sector handles extremely sensitive information that should never leave a professional's computer. LegalLocal RAG is a document analysis tool that works 100% offline, with no external API calls and no internet connection required.

## Why this approach?

Most current artificial intelligence systems (like ChatGPT) send your data to external servers. For a lawyer, this is an unacceptable risk. This system runs entirely on your machine, leveraging language models optimized to work on standard office laptops.

### The Problem with Traditional Text Splitting (Chunking)

When you try to have an AI analyze a long contract, the system must divide the text into small pieces. In a traditional RAG system, these cuts are arbitrary and can split an important clause right in the middle, causing the AI to lose the context of an exception or a penalty.

To solve this, we implemented a strategy called **Parent-Child Indexing**:
- **Precise Search (Child)**: The system looks through small fragments to find exactly where your answer might be.
- **Full Context (Parent)**: Once the exact point is found, the system hands the full clause or article to the AI. This way, the response always considers the complete legal context.

## Technical Choices

I have selected these tools to find the best balance between speed and precision on systems without a dedicated graphics card (pure CPU):

*   **Qwen 3 4B (GGUF)**: This is the brain of the system. I chose it because it is surprisingly fast on normal processors and understands the structure of legal documents very well.
*   **BGE-small-en-v1.5**: A very lightweight embedding model (only 130MB) that allows for almost instantaneous information retrieval.
*   **Chainlit**: For the interface, instead of something overly complex, I used Chainlit because it offers a modern chat experience and allows you to view the original source text in a side panel by clicking on references.

## Quick Setup

### 1. Prepare the Environment
You need Python 3.10 or higher. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. The Language Model
Download the `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` file and place it inside the `/models` folder. The system will automatically detect it when it starts.

### 3. Running the App
To start working, run the following command:

```bash
chainlit run app.py -w
```

The application will open in your browser (usually at `http://localhost:8000`).

## Notes on Hardware and Privacy

This software has been designed to work on laptops with i5 or Ryzen 5 processors and 16GB of RAM. You do not need a powerful graphics card.

**About your privacy:** There is no telemetry, no analytics, and no cloud connection. What happens in LegalLocal RAG stays on your computer.

---
*Note: This tool is a research assistant and does not replace the legal judgment of a professional. Always verify answers with the original document through the side references.*
