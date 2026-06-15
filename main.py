from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4
import os

app = FastAPI(title="AgentOS Production API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "agentos-secret")
ALGORITHM = "HS256"

security = HTTPBearer()

users = {}

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

def create_token(email: str):
    payload = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        email = payload.get("sub")

        if email not in users:
            raise HTTPException(status_code=401)

        return users[email]

    except:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "AgentOS AI v2"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/api/register")
def register(data: RegisterRequest):

    if data.email in users:
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    users[data.email] = {
        "email": data.email,
        "password": data.password,
        "name": data.name
    }

    return {
        "success": True,
        "message": "User created"
    }

@app.post("/api/login")
def login(data: LoginRequest):

    user = users.get(data.email)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    if user["password"] != data.password:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_token(data.email)

    return {
        "access_token": token,
        "user": {
            "email": user["email"],
            "name": user["name"]
        }
    }

@app.get("/api/me")
def me(user=Depends(get_current_user)):
    return user
class ChatCreate(BaseModel):
    title: str = "New Chat"

class ChatMessage(BaseModel):
    role: str
    content: str

chats_db = {}

@app.post("/api/chats")
def create_chat(chat: ChatCreate):

    chat_id = str(uuid4())

    chats_db[chat_id] = {
        "id": chat_id,
        "title": chat.title,
        "messages": []
    }

    return chats_db[chat_id]


@app.get("/api/chats")
def get_chats():
    return list(chats_db.values())
