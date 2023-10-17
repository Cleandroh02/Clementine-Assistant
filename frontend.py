import streamlit as st
from pathlib import Path
import time
from src.rag.rag import RagApp


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


st.title("Clementine-Bot")
if "rag" not in st.session_state:
    st.session_state.rag = RagApp()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hi, how can I help you?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        rag_res = st.session_state.rag.answer_question(prompt)
        for chunk in rag_res["result"].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": rag_res["result"]}
        )

    docs_about = rag_res["source_documents"]
    if docs_about:
        st.header(
            "The following are some related articles to your question..."
            )
        all_paths = []
        titles = []
        for doc in docs_about:
            path_ = doc.metadata["source"]
            with st.expander(path_.split("/")[-1][:-2]):
                markdown_ = read_markdown_file(path_)
                st.markdown(markdown_, unsafe_allow_html=True)
