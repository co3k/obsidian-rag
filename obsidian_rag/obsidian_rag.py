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
import os

from dotenv import load_dotenv

load_dotenv()

loader = ObsidianLoader(os.getenv("PATH_TO_OBSIDIAN_VOLUME"))
docs = loader.load()

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)

memory = ConversationBufferMemory()

prompt_template = PromptTemplate.from_template("{input}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
question_generator_chain = LLMChain(llm=llm, prompt=prompt_template)
combine_docs_chain = StuffDocumentsChain(llm_chain=question_generator_chain)

history_aware_retriever = create_history_aware_retriever(
    llm, vector_store.as_retriever(), prompt_template,
)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_generator_chain,
)

st.title("Obsidian から検索")
prompt = st.text_input("質問を入力してください")

if prompt:
    response = rag_chain.invoke({"input": prompt, "chat_history": memory.buffer})
    st.write(response["answer"]["text"])