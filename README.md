# RAG Playground

A Streamlit application to explore different chunking and retrieval strategies for Retrieval-Augmented Generation (RAG) using LangChain.

## Features

- Upload PDF and text documents
- Experiment with various text chunking strategies:
  - RecursiveCharacterTextSplitter
  - CharacterTextSplitter
  - TokenTextSplitter
  - SentenceTransformersTokenTextSplitter
  - MarkdownHeaderTextSplitter
- Choose between different vector stores:
  - FAISS
  - Chroma
- Try different retrieval methods:
  - Basic retriever with similarity search
  - Basic retriever with MMR search
  - Contextual compression retriever
- Visualize chunks and their statistics
- Compare results across different configurations

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rag_playground
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key (copy from `.env.example`):
```bash
cp .env.example .env
```
Then edit the `.env` file to include your API key.

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your documents (PDF or text)

3. Configure your chunking and retrieval strategy

4. Process the documents

5. Ask questions and explore results

## Project Structure

- `app.py`: Main Streamlit application
- `utils/text_splitters.py`: Text splitter implementations
- `utils/retrievers.py`: Retriever implementations
- `requirements.txt`: Required dependencies

## How to Use the Comparison Feature

The comparison feature allows you to compare different chunking and retrieval strategies:

1. Configure and process your documents with one strategy
2. Ask questions
3. Change configuration and process again
4. Ask the same questions
5. Add both sets of results to the comparison table
6. Analyze differences in answers, retrieved documents, and response times

## Future Extensions

- Add support for more document types (HTML, DOCX, etc.)
- Include more vector databases (Pinecone, Weaviate, etc.)
- Add more advanced retrieval techniques
- Support hybrid search methods 