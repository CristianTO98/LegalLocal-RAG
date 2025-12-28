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

## Optimización de Rendimiento: Precisión de Tokens

Este proyecto ha evolucionado de medir caracteres a utilizar **Tokens reales** para garantizar un rendimiento óptimo y predecible en CPU:

1. **Ingeniería de Contexto con tiktoken**:
    *   **La Lógica**: Los caracteres son engañosos (un carácter en español no ocupa lo mismo que en latín o código). Al usar `tiktoken`, medimos exactamente lo que la IA "ve".
    *   **El Cambio**: El sistema ahora fragmenta los documentos basándose en el conteo de tokens, no de letras.

2. **Filtrado Estricto y Contexto Ágil (2 fuentes x 550 tokens)**:
    *   **La Lógica**: Para ganar velocidad sin sacrificar veracidad, hemos endurecido el umbral de similitud (Distancia Coseno de 0.6 a **0.45**).
    *   **El Cambio**: Ahora solo se envían las **2 mejores fuentes** que superen este filtro de calidad.
    *   **El Porqué**: Al reducir de 3 a 2 fuentes, el volumen de tokens baja drásticamente (~1100 tokens de contexto). Esto libera espacio en el `n_batch` de 2048, permitiendo que la respuesta se genere aún más rápido y con fuentes de mayor relevancia semántica. El progreso y el tamaño del contexto se pueden monitorear en tiempo real en la consola (medido en tokens).

3. **Fragmentación de Alta Resolución**:
    *   **La Lógica**: Chunks hijos de 150 tokens con un solapamiento de 50.
    *   **El Porqué**: Al usar tokens para el solapamiento, nos aseguramos de que no se pierda el hilo semántico entre fragmentos, independientemente del idioma del documento.

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
