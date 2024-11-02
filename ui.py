import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os
import shutil
import time
from rag import ChatPDF
from dotenv import load_dotenv

st.set_page_config(page_title="ChatPDF")
load_dotenv("./.env")


def read_and_save_file():
    if "assistant" not in st.session_state:
        if st.session_state["local_key"]:
            st.session_state.assistant = ChatPDF(
                st.session_state["groq_key_local"], st.session_state["hf_key_local"])
        else:
            if not st.session_state["groq_key"]:
                st.warning(
                    "Please add your **Groq API key** to continue.")
                return

            if not st.session_state["hf_key"]:
                st.warning(
                    "Please add your **Huggingface API key** to continue.")
                return
    st.session_state.messages = []
    if len(st.session_state["file_uploader"]) == 0:
        st.warning(
            "Please upload your PDFs to continue.")
        return

    temp_dir_path = tempfile.mkdtemp()
    for uploaded_file in st.session_state["file_uploader"]:
        file_path = os.path.join(temp_dir_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting PDFs..."):
        st.session_state["assistant"].ingest(temp_dir_path)

    shutil.rmtree(temp_dir_path)
    st.success("Ingest all PDFs successfully!")


def page():
    with st.sidebar:
        st.markdown("### some introduction to ChatPDF")
        st.text_input(
            "Groq API Key", key="groq_key", type="password")
        st.text_input(
            "Huggingface API Key", key="hf_key", type="password")

    if os.getenv("GROQ_API_KEY") and os.getenv("HF_TOKEN"):
        st.session_state["groq_key_local"] = os.getenv("GROQ_API_KEY")
        st.session_state["hf_key_local"] = os.getenv("HF_TOKEN")
        st.session_state["local_key"] = True
    else:
        st.session_state["local_key"] = False

    st.session_state["notice"] = st.empty()
    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
    st.session_state["ingestion_spinner"] = st.empty()

    st.chat_message("assistant").write("How can I help you?")

    if prompt := st.chat_input():
        if not st.session_state["local_key"]:
            if not st.session_state["groq_key"]:
                st.session_state["notice"].warning(
                    "Please add your **Groq API key** to continue.")
                st.stop()

            if not st.session_state["hf_key"]:
                st.session_state["notice"].warning(
                    "Please add your **Huggingface API key** to continue.")
                st.stop()

        if not st.session_state["file_uploader"]:
            st.session_state["notice"].warning(
                "Please upload your PDFs to continue.")
            st.stop()

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        st.session_state["thinking_spinner"] = st.empty()
        answer = None
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            answer = st.session_state.assistant.ask(prompt)["answer"]
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    page()
