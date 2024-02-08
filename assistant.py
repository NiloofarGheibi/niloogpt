import os
import sys
from typing import Optional, Tuple

import gradio
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

import constants

def new_chain():
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", n=3)
    system_template = SystemMessagePromptTemplate.from_template("You are an expert e-commerce, marketplace business and fashion industry. Your main area of expertise and responsibility is to guide Fashion Brands and Merchants to maximize their growth and profit, as well as return rate. You are detail-oriented and can crunch data and bring meaning to them very well. Any instruction and recommendation you give to the merchants and sellers are actionable by them. Don't use cheesy and buzz words, and stick to facts. Keep the answers short and concise, and on point. Avoid using keywords like significant, etc, but meaningful text. In case you could not recommend it, ask the user to reach out to their partner consultants to get guidance. Split the output into 3 sections: summarizing highlights and lowlights along with at least 3 suggestions to make their profit margin bigger.")
    user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
    template = ChatPromptTemplate.from_messages([system_template, user_template])
    loader = DirectoryLoader("data/")
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=1))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    chain =  LLMChain(llm=llm, prompt=template, verbose=True, memory=memory)
    return chain

def set_openai_api_key():
    os.environ["OPENAI_API_KEY"] = constants.APIKEY
    return new_chain()

def CustomAssistant(prompt):
    set_openai_api_key()
    chain = new_chain()
    chat_history = []
    if prompt:
        text = chain({"user_prompt": prompt})
        return text['text']
    return "Type a prompt to get guidance!"


demo = gradio.Interface(fn=CustomAssistant, inputs = "text", outputs = "text", title = "E-Commerce Assistant")

demo.launch(share=True)
