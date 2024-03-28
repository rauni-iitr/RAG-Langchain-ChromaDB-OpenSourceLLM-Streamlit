pypdf
langchain
transformers
chromadb
streamlit
sentence-transformers
# Example: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir



# Enviornment Setup 

1. Clone the repo using git:
    ```shell
    git clone https://github.com/rauni-iitr/langchain_chromaDB_opensourceLLM_streamlit.git
    ```

2. Create a virtual enviornment, with 'venv' or with 'conda' and activate.
    ```shell
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Now this rag application is built using few dependencies:
    - pypdf -- for reading pdf documents
    - chromadb -- vectorDB for creating a vector store
    - transformers -- dependency for sentence-transfors, atleast in this repository
    - sentence-transformers -- for embedding models to convert pdf documnts into vectors
    - streamlit -- to make UI for the LLM PDF's Q&A
    - llama-cpp_python -- to load gguf files for CPU inference of LLMs

    You can install all of these with pip;
    ```shell
    pip install pypdf chromadb transformers sentence-transformers streamlit
    ```
    
 