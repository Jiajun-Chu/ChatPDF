# ChatPDF

ChatPDF is a text-based conversational tool built on Retrieval-Augmented Generation (**RAG**). Users can upload multiple **PDFs**, and retrieve document knowledge through conversational queries. The tool also remembers **chat history**, enabling multi-turn conversations.

## Project Overview

This is an entry-level practice project aimed at understanding the basic workflow of RAG and familiarizing with common machine learning tools such as **HuggingFace, LangChain, and Chroma**. Below is a basic workflow diagram of ChatPDF:

<Insert workflow diagram here>

## Depolyment

1. **Import API Keys**: Create a `.env` file with your keys for automatic access, or manually enter them on the interface (see figure 2).

2. **Install Dependencies**: Execute `pip install -r requirements.txt` and ensure all required libraries are installed.

3. **Run Script**: Execute `streamlit run ui.py` in the terminal. This will open the interface (figure 2).

4. **Upload PDFs**: Add your PDFs via the interface.

5. **Query**: Once ingestion is complete, start querying the documents.

## Configuration

- **Embedding Model**: Utilizes the open-source `models--BAAI--bge-large-en-v1.5`. Register on **HuggingFace** and generate a private key.

- **LLM Model**: Uses **Groq**'s free API for the `llama3-8b-8192` model. Register on Groq and obtain a private key. Note: Groq's free API may be unstable; consider using a paid alternative.

- **Vector Database**: Employs **Chroma** for storing and retrieving document vectors. Chroma efficiently handles embeddings, providing quick and accurate query responses.

## To be improved

1. Upgrade to superior embedding and LLM models.

2. Integrate rerank, fusion, and other advanced RAG techniques

3. Evaluate the performance of RAG system

4. Cloud Deployment
