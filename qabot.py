from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


def get_llm():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm














