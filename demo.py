import os
import sys
from typing import Optional, Tuple

import gradio
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma

import constants


def load_chain():
    loader = DirectoryLoader("data/")
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model="gpt-3.5-turbo"),
      retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    return chain


def set_openai_api_key():
    os.environ["OPENAI_API_KEY"] = constants.APIKEY
    return load_chain()

def CustomBot(prompt):
    set_openai_api_key()
    chain = load_chain()
    chat_history = []
    if prompt:
        text = chain({"question": prompt, "chat_history": chat_history})
        chat_history.append((prompt, text['answer']))
        return text['answer']
    return "Type a prompt to get guidance!"

demo = gradio.Interface(fn=CustomBot, inputs = "text", outputs = "text", title = "<NAME YOUR PRO ASSISTANT>")

demo.launch(share=True)
