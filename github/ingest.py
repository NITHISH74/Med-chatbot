import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated FAISS import


embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
DB_FAISS_PATH = 'db_faiss'

loader = DirectoryLoader('data/', glob="**/*.pdf",show_progress=True, loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

texts = text_splitter.split_documents(documents)


db = FAISS.from_documents(
    texts,
    embeddings
)
db.save_local(DB_FAISS_PATH)


print("Vector DB Successfully Created!")

