from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers, LlamaCpp
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

import os
import json
import string


from pydantic.v1 import BaseModel
app = FastAPI(debug=True)
DB_FAISS_PATH = 'db_faiss'


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q8_0.gguf"

# Make sure the model path is correct for your system!
llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")



print("Loading FAISS database...")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
print("FAISS database loaded successfully")

print("LLM initialized successfully")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    result = qa_chain.run(question)
    return templates.TemplateResponse("index.html", {"request": request, "question": question, "answer": result})

if __name__ == "__main__":
    print("App is running...")

#uvicorn app02:app --host 127.0.0.1 --port 8000 --reload
    