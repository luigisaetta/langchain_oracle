import streamlit as st

# for pdf post processing
import re

# modified to load from Pdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# two possible vector store
from langchain.vectorstores import Chroma

from langchain.schema.runnable import RunnablePassthrough

# removed OpenAI, using Cohere embeddings
from langchain.embeddings import CohereEmbeddings

from langchain import hub

import oci

# oci_llm is in a local file
from oci_llm import OCIGenAILLM

# config for the RAG
from config_rag import BOOK, CHUNK_SIZE, CHUNK_OVERLAP, MAX_TOKENS, ENDPOINT

# private configs
from config_private import COMPARTMENT_OCID, COHERE_API_KEY

DEBUG = False

CONFIG_PROFILE = "DEFAULT"


#
# def load_oci_config()
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print(oci_config)

    return oci_config


#
# def: Initialize_rag_chain
#
# to run it only once
@st.cache_resource
def initialize_rag_chain():
    # Initialize RAG

    # for oci llm
    oci_config = load_oci_config()

    # Loading the pdf document
    loader = PyPDFLoader(BOOK)

    docs = loader.load()

    print("PDF document loaded!")

    # split in chunks
    # try with smaller chuncks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    splits = text_splitter.split_documents(docs)

    print(f"We have splitted the pdf in {len(splits)} splits...")

    # some post processing
    for split in splits:
        split.page_content = split.page_content.replace("\n", " ")
        split.page_content = re.sub("[^a-zA-Z0-9 \n\.]", " ", split.page_content)
        # remove duplicate blank
        split.page_content = " ".join(split.page_content.split())

    print("Initializing vector store...")
    cohere = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

    # using Chroma as Vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=cohere)

    # increased num. of docs to 5 (default to 4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Build the class for OCI GenAI
    llm = OCIGenAILLM(
        max_tokens=MAX_TOKENS,
        config=oci_config,
        compartment_id=COMPARTMENT_OCID,
        endpoint=ENDPOINT,
        debug=DEBUG,
    )

    # for now hard coded...
    rag_prompt = hub.pull("rlm/rag-prompt")

    print("Building rag_chain...")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
    )

    return rag_chain


#
# def: get_answer  from LLM
#
def get_answer(rag_chain, question):
    response = rag_chain.invoke(question)

    print(f"Question: {question}")
    print("The response:")
    print(response)
    print()

    return response
