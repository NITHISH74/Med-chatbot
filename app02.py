from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers, LlamaCpp
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

import os
import json
import string


from pydantic.v1 import BaseModel

app = FastAPI(debug=True)
DB_FAISS_PATH = 'db_faiss'

USERNAME = "user"
PASSWORD = "password"
SECRET_KEY = "your_secret_key"

from itsdangerous import URLSafeSerializer
serializer = URLSafeSerializer(SECRET_KEY)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q8_0.gguf"

# Make sure the model path is correct for your system!
llm = CTransformers(
        model = "BioMistral-7B.Q8_0.gguf",
        max_new_tokens = 512,
        temperature = 0.4
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

# Function to get session cookie
def get_session_cookie(request: Request):
    session_cookie = request.cookies.get("session")
    if session_cookie:
        try:
            return serializer.loads(session_cookie)
        except Exception:
            return None
    return None

# Login Page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Handle Login
@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == USERNAME and password == PASSWORD:
        session_token = serializer.dumps({"username": username})
        response = RedirectResponse(url="/chatbot", status_code=303)
        response.set_cookie(key="session", value=session_token)
        return response
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

# Chatbot Page (index)
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    session = get_session_cookie(request)
    if session is None:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request})

# Ask Question
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    session = get_session_cookie(request)
    if session is None:
        return RedirectResponse(url="/login")
    result = qa_chain.run(question)
    return templates.TemplateResponse("index.html", {"request": request, "question": question, "answer": result})

if __name__ == "__main__":
    print("App is running...")

#uvicorn app02:app --host 127.0.0.1 --port 8000 --reload
    