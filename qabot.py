from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    
    return HuggingFacePipeline(pipeline=pipe)


def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document
    

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def hf_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def vector_database(chunks):
    embedding_model = hf_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever


def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )

    response = qa.invoke(query)
    return response['result']


#interface

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF QA Bot (Hugging Face)",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


if __name__ == "__main__":
    rag_application.launch(share=True)















