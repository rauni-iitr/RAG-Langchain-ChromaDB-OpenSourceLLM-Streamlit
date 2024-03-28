import os
import chromadb
import streamlit as st

from langchain_community.document_loaders import HuggingFaceDatasetLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# for streaming in Streamlit without LECL
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

####################### Data processing for vectorstore #################################
pdf_folder_path = "./data_source"
documents = []

for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path,file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

text_splitter_rc = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents_rc = text_splitter_rc.split_documents(documents)

####################### EMBEDDINGS #################################
model_path = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
persist_directory="./vector_stores"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

####################### RAG #################################


prompt_template = """Use the following pieces of context regarding titanic ship to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

#VectorDB creation and saving to disk
client = chromadb.Client()

persist_directory="/Users/raunakanand/Documents/Work_R/llm0/vector_stores"
vectordb = Chroma.from_documents(
    documents=chunked_documents_rc,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name='chroma1'
)
vectordb.persist()

#VectorDB -loading from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name='chroma1')
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


n_gpu_layers = 1
n_batch = 512
# stream_handler = StreamHandler(st.empty())

llm = LlamaCpp(
    model_path="/Users/raunakanand/Documents/Work_R/llm_models/mistral-7b-v0.1.Q4_K_S.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=1024,
    f16_kv=True,
    verbose=True,
    # streaming=True,
    # callbacks=[stream_handler]
    # callbacks=[StreamingStdOutCallbackHandler()]
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                 retriever=retriever,
                                #  return_source_documents=True,
                                 chain_type_kwargs={'prompt': prompt},
                                 verbose=False)

rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | 
             prompt | llm | StrOutputParser())

def inference(query: str):
    # return qa.invoke(query)['result']
    # return qa.run(query)
    return rag_chain.stream(query)

print('final')



