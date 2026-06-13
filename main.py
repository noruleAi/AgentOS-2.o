import os, json, uuid, datetime, logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, EmailStr
import asyncpg
import bcrypt
import jwt
from jwt.exceptions import InvalidTokenError
import httpx
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiofiles

# ============== Logging ==============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentos")

# ============== Configuration ==============
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://agentos:agentos@db:5432/agentos")
SECRET_KEY = os.getenv("SECRET_KEY")                           # MUST be set in environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",") # e.g., "https://agentos-ai.in"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# ============== Rate Limiter ==============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Database Pool ==============
pool = None

async def get_db():
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(DATABASE_URL)
    async with pool.acquire() as conn:
        yield conn

# ============== Auth Helpers ==============
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_access_token(user_id: int, email: str, role: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "email": email, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def get_current_user(request: Request, db=Depends(get_db)):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth.split(" ")[1]
    try:
        payload = decode_token(token)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = int(payload["sub"])
    user = await db.fetchrow("SELECT id, email, name, role, status FROM users WHERE id=$1", user_id)
    if not user or user["status"] != "active":
        raise HTTPException(status_code=401, detail="User inactive or not found")
    return user

def founder_required(user=Depends(get_current_user)):
    if user["role"] != "founder":
        raise HTTPException(status_code=403, detail="Founder only")
    return user

# ============== Schemas ==============
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatCreate(BaseModel):
    title: str = "New Chat"

class ChatStreamRequest(BaseModel):
    chat_id: str
    messages: List[dict]
    model: str = "openai/gpt-3.5-turbo"

# ============== Startup ==============
@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS chats (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT CHECK(role IN ('user','assistant','system')) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
        """)
    logger.info("Database initialized")

# ============== Health ==============
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

# ============== Auth Endpoints ==============
@app.post("/api/register")
async def register(user: UserRegister, db=Depends(get_db)):
    try:
        hashed = hash_password(user.password)
        role = "founder" if user.email in os.getenv("FOUNDER_EMAILS", "").split(",") else "user"
        await db.execute(
            "INSERT INTO users (email, name, password_hash, role) VALUES ($1,$2,$3,$4)",
            user.email, user.name, hashed, role
        )
        logger.info(f"User registered: {user.email}")
        return {"message": "User registered"}
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=400, detail="Email already exists")

@app.post("/api/login")
async def login(user: UserLogin, db=Depends(get_db)):
    row = await db.fetchrow("SELECT * FROM users WHERE email=$1", user.email)
    if not row or not verify_password(user.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if row["status"] != "active":
        raise HTTPException(status_code=403, detail="Account deactivated")
    token = create_access_token(row["id"], row["email"], row["role"])
    logger.info(f"User logged in: {user.email}")
    return {
        "access_token": token,
        "user": {"id": row["id"], "email": row["email"], "name": row["name"], "role": row["role"]}
    }

# ============== Chat Endpoints ==============
@app.get("/api/chats")
async def get_chats(user=Depends(get_current_user), db=Depends(get_db)):
    rows = await db.fetch("SELECT id, title, created_at FROM chats WHERE user_id=$1 ORDER BY created_at DESC", user["id"])
    return [dict(r) for r in rows]

@app.post("/api/chats")
async def create_chat(chat: ChatCreate, user=Depends(get_current_user), db=Depends(get_db)):
    chat_id = str(uuid.uuid4())
    await db.execute("INSERT INTO chats (id, user_id, title) VALUES ($1,$2,$3)", chat_id, user["id"], chat.title)
    return {"id": chat_id, "title": chat.title, "created_at": datetime.datetime.utcnow().isoformat()}

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, user=Depends(get_current_user), db=Depends(get_db)):
    chat = await db.fetchrow("SELECT * FROM chats WHERE id=$1 AND user_id=$2", chat_id, user["id"])
    if not chat:
        raise HTTPException(status_code=404, detail="Not found")
    msgs = await db.fetch("SELECT role, content, created_at FROM messages WHERE chat_id=$1 ORDER BY created_at", chat_id)
    return {"id": chat["id"], "title": chat["title"], "messages": [dict(m) for m in msgs]}

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user=Depends(get_current_user), db=Depends(get_db)):
    await db.execute("DELETE FROM chats WHERE id=$1 AND user_id=$2", chat_id, user["id"])
    return {"status": "deleted"}

# ============== Streaming (OpenRouter) ==============
async def stream_openrouter(messages: list, model: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AgentOS",
            },
            json={"model": model, "messages": messages, "stream": True},
            timeout=60.0
        ) as response:
            full = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except:
                        pass
            yield "data: [DONE]\n\n"
    return full

@app.post("/api/chat/stream")
@limiter.limit("20/minute")
async def stream_chat(request: Request, req: ChatStreamRequest, user=Depends(get_current_user), db=Depends(get_db)):
    chat = await db.fetchrow("SELECT id FROM chats WHERE id=$1 AND user_id=$2", req.chat_id, user["id"])
    if not chat:
        raise HTTPException(status_code=403, detail="Not your chat")

    # Save user message
    if req.messages and req.messages[-1]["role"] == "user":
        await db.execute("INSERT INTO messages (id, chat_id, role, content) VALUES ($1,$2,$3,$4)",
                         str(uuid.uuid4()), req.chat_id, "user", req.messages[-1]["content"])

    async def save_and_stream():
        full_response = ""
        async for chunk in stream_openrouter(req.messages, req.model):
            if chunk.startswith("data: [DONE]"):
                break
            yield chunk
            try:
                data = json.loads(chunk.replace("data: ", ""))
                full_response += data.get("content", "")
            except:
                pass
        if full_response:
            await db.execute("INSERT INTO messages (id, chat_id, role, content) VALUES ($1,$2,$3,$4)",
                             str(uuid.uuid4()), req.chat_id, "assistant", full_response)
        yield "data: [DONE]\n\n"

    return StreamingResponse(save_and_stream(), media_type="text/event-stream")

# ============== Simple Chat (backward compat) ==============
@app.get("/chat")
@limiter.limit("30/minute")
async def simple_chat(request: Request, message: str):
    if not OPENROUTER_API_KEY:
        return {"reply": "OpenRouter API key not configured."}
    try:
        resp = await httpx.AsyncClient().post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message}],
            },
            timeout=30.0
        )
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        logger.error(f"Simple chat error: {e}")
        return {"reply": f"Error: {str(e)}"}

# ============== File Upload Validation & Face/Video ==============
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_file(file: UploadFile):
    if file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    # Additional checks can be added here (mime type, extension)

@app.post("/api/face-search")
async def face_search(file: UploadFile = File(...), user=Depends(get_current_user)):
    await validate_file(file)
    # TODO: integrate real face recognition (e.g., AWS Rekognition or Gemini Vision)
    logger.info(f"Face search requested by {user['email']} for file {file.filename}")
    return {"result": f"Face processed (placeholder): {file.filename}"}

@app.post("/api/video-analyze")
async def video_analyze(file: UploadFile = File(...), user=Depends(get_current_user)):
    await validate_file(file)
    logger.info(f"Video analysis requested by {user['email']} for file {file.filename}")
    return {"summary": f"Video analysis not yet implemented: {file.filename}"}

# ============== Frontend Serve (optional, for convenience) ==============
@app.get("/")
async def home():
    return {"message": "AgentOS AI Backend Running"}