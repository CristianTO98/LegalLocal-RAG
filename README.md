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

## Design Decisions: Quality vs. Speed (RAG Tuning)

This project has undergone multiple performance tests. While it is possible to achieve **38-second** responses with faster, smaller contexts, in the legal sector, **precision (Recall)** is non-negotiable. I have opted for a "Slow but Sure" configuration, where wait times typically range up to **60 seconds** to ensure maximum reliability and comprehensive context.

### Proven Results
*   **Zero Hallucinations**: Verified through extensive **human evaluation**, the system consistently avoids "hallucinations" by enforcing strict context-only rules and high-integrity chunking, staying 100% faithful to the source text.
*   **High Precision on Complex Queries**: Human testers have confirmed the system successfully handles difficult legal questions where standard RAG systems fail, such as identifying specific exceptions or multi-part procedural requirements.

1. **Broad Retrieval (4 Sources & 0.6 Threshold)**:
    *   **Why**: Legal answers are often buried. By retrieving up to 4 sources with a 0.6 similarity threshold, we capture critical exceptions that often appear in the 3rd or 4th ranking position.
    *   **Impact**: It is slower, but it drastically reduces the risk of omitting critical legal information.

2. **Optimized Chunking (Parent: 550 tokens / Child: 150 tokens)**:
    *   **Why**: These sizes are the "sweet spot" for performance and context. 550-token parent blocks ensure the LLM sees full articles, while 150-token child blocks allow for pinpoint search accuracy.
    *   **Result**: This configuration yielded the **best overall results** during testing, providing the perfect balance between granular retrieval and rich legal context. It ensures the AI maintains a global view of the legal norm, preventing fragmented or out-of-context answers.

3. **Intelligent Context Assembly (Search & Deduplication)**:
    *   **The Strategy**: The system performs a semantic search on the smaller "Child" chunks to find relevant information. However, instead of passing those small fragments to the AI, it retrieves the full "Parent" section they belong to.
    *   **Deduplication**: To ensure high-quality information without redundancy, the system tracks which parent sections have already been selected. If multiple search results point to the same clause or article, it is only included once in the context, maximizing the diversity of the information provided to the LLM.

4. **Batch Processing (n_batch: 2048)**:
    *   **Why**: To handle nearly 2,000 tokens of context on a CPU, the engine uses large batches. This leverages the processor's maximum mathematical capacity (AVX instructions), mitigating the delay of high data volume.

## Continuous Evaluation

To ensure the reliability of the system over time, we can integrate **Ragas** (Retrieval Augmented Generation Assessment). This allows for:
*   **Evaluation in Production**: Continuous monitoring of faithfulness and relevancy of the answers.
*   **Metric-driven Tuning**: Using specialized metrics to further refine chunk sizes and retrieval thresholds.

### Benchmarked Hardware
Performance data was verified on the following system:
*   **Processor**: AMD Ryzen 5 4500U with Radeon Graphics (2.38 GHz)
*   **RAM**: 16.0 GB (15.4 GB usable)
*   **System Type**: 64-bit Operating System, x64-based processor

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

You can find the model on Hugging Face (e.g., Qwen/Qwen3-4B-Instruct-GGUF).

### 3. Running the App
The project includes a sample legal document (`samples/fcra-may2023-508.pdf`) so you can start testing immediately. To start the app, run:

```bash
chainlit run app.py -w
```

The application will open in your browser (usually at `http://localhost:8000`). Upload the PDF from the `samples/` folder to begin.

## Notes on Hardware and Privacy

This software has been designed to work on laptops with i5 or Ryzen 5 processors and 16GB of RAM. You do not need a powerful graphics card.

**About your privacy:** There is no telemetry, no analytics, and no cloud connection. What happens in LegalLocal RAG stays on your computer.

---
*Note: This tool is a research assistant and does not replace the legal judgment of a professional. Always verify answers with the original document through the side references.*
