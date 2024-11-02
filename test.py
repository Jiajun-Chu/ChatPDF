from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from rag import ChatPDF
from dotenv import load_dotenv
import os
load_dotenv()

rag = ChatPDF(
    os.getenv("GROQ_API_KEY"), os.getenv("HF_TOKEN")
)
rag.ingest("./data")
rag.ingest("./data")
rag.ingest("./data")
print(rag.vectordb_current)
rag.ingest("./data")
