import streamlit as st
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
    ConversationalRetrievalChain,
    LLMChain,
    StuffDocumentsChain,
)
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse()

if "langfuse_handler" not in st.session_state:
    st.session_state["langfuse_handler"] = CallbackHandler(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host="https://us.cloud.langfuse.com",
        session_id="session_id",
    )
langfuse_handler = st.session_state["langfuse_handler"]

if "rag_chain" not in st.session_state:
    loader = ObsidianLoader(os.getenv("PATH_TO_OBSIDIAN_VOLUME"))
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)

    prompt_template = PromptTemplate.from_template("{input}")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    question_generator_chain = LLMChain(llm=llm, prompt=prompt_template)
    combine_docs_chain = StuffDocumentsChain(llm_chain=question_generator_chain)

    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), prompt_template,
    )

    st.session_state["rag_chain"] = create_retrieval_chain(
        history_aware_retriever, question_generator_chain,
    )
rag_chain = st.session_state["rag_chain"]

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()
memory = st.session_state["memory"]

st.title("Obsidian から検索")
prompt = st.text_input("質問を入力してください")

if prompt:
    response = rag_chain.invoke(
        {"input": prompt, "chat_history": memory.buffer},
        config={"callbacks": [langfuse_handler]},
    )
    st.write(response["answer"]["text"])