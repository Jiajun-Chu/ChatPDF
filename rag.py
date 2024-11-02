# install all dependencies with command:
# pip install -r requirements.txt

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import shutil
import tempfile
from langchain.retrievers import (
    MergerRetriever,
)


class ChatPDF:
    def __init__(self, groq_key, hf_key):
        self.groq_key = groq_key
        self.hf_key = hf_key

        # self.embedding_models = ['BAAI/bge-large-en-v1.5']
        # self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_models[0],cache_folder="./cache")

        self.embeddings_dir = "./cache/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
        # "./cache/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        # self.embeddings_dir = "./cache/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        self.embedding_models = ['BAAI/bge-large-en-v1.5']
        if os.path.exists(self.embeddings_dir):
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_dir)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_models[0], cache_folder="./cache")
        self.vectordb = None
        self.retriever = None
        self.prompts = self.load_prompts("./prompts.json")
        self.q_prompt = None
        self.qa_prompt = None
        self.history_aware_retriever = None
        self.qa_chain = None
        self.rag_chain = None
        self.store = {}
        self.conversational_rag_chain = None

        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.1,
            max_retries=2,
            groq_api_key=self.groq_key,
        )

        self.vectordb_previous = {}
        self.vectordb_current = {}

    def load_prompts(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def retriever_for_each_pdf(self, pdf_file_name, pdf_file_path):
        loader = PyPDFLoader(os.path.join(pdf_file_path, pdf_file_name))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = loader.load_and_split(text_splitter=text_splitter)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        retriever = vectordb.as_retriever()
        return pdf_file_name, retriever

    def ingest(self, pdf_file_path):
        pdf_files = [file for file in os.listdir(
            pdf_file_path) if file.endswith('.pdf')]

        # loader = DirectoryLoader(pdf_file_path, loader_cls=PyPDFLoader)
        self.vectordb_previous = self.vectordb_current
        self.vectordb_current = {}
        for pdf_name_current in pdf_files:
            for pdf_name_previous, retriever in self.vectordb_previous.items():
                if (pdf_name_previous == pdf_name_current):
                    self.vectordb_current[pdf_name_previous] = retriever
                    break
            if pdf_name_current not in self.vectordb_current.keys():
                temp_pdf_name, temp_retriever = self.retriever_for_each_pdf(
                    pdf_name_current, pdf_file_path)
                self.vectordb_current[temp_pdf_name] = temp_retriever
                temp_pdf_name = None
                temp_retriever = None

        retriever_list = list(self.vectordb_current.values())
        self.merger = MergerRetriever(retrievers=retriever_list)

        self.q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["q_system_prompt"]),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.merger, self.q_prompt
        )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts["qa_system_prompt"]),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.qa_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, query: str):
        if not self.conversational_rag_chain:
            return "Please add a  PDF document first."
        result = self.conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
        )
        return result
