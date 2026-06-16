import os, json, uuid, datetime, logging, secrets, base64, hashlib, sqlite3, urllib.parse
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, constr
from jose import jwt, JWTError
import httpx
from passlib.context import CryptContext

# ========================= CONFIGURATION =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "agentos-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
FOUNDER_EMAILS = set(
    email.strip().lower()
    for email in os.getenv("FOUNDER_EMAILS", "").split(",")
    if email.strip()
)

# Founder credentials (environment variables or fallback for local testing)
FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "rahulkumarmahto2024@gmail.com").lower()
FOUNDER_PASSWORD = os.getenv("FOUNDER_PASSWORD", "12345678")  # Change in production!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentos")

# ========================= Password Hashing =========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

# ========================= SQLite Database =========================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentos.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")   # ✅ Enable foreign keys
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            status TEXT DEFAULT 'active',
            is_verified INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'New Chat',
            pinned INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY(user_email) REFERENCES users(email)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'default',
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            UNIQUE(user_email, profile, key),
            FOREIGN KEY(user_email) REFERENCES users(email)
        )
    """)
    conn.commit()
    conn.close()

# ========================= FastAPI App =========================
app = FastAPI(title="AgentOS AI 2.0", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = BASE_DIR
IMAGES_DIR = os.path.join(BASE_DIR, "images")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

security = HTTPBearer()

def create_access_token(email: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = decode_token(token)
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND status='active'", (email,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    return dict(user)

def founder_required(current_user=Depends(get_current_user)):
    if current_user.get("role") != "founder":
        raise HTTPException(status_code=403, detail="Founder only")
    return current_user

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("AgentOS AI 2.0 starting up")
    logger.info(f"OPENROUTER_API_KEY configured: {bool(OPENROUTER_API_KEY)}")
    logger.info(f"GEMINI_API_KEY configured: {bool(GEMINI_API_KEY)}")
    # Ensure founder exists (using env vars)
    conn = get_db()
    if not conn.execute("SELECT 1 FROM users WHERE email=?", (FOUNDER_EMAIL,)).fetchone():
        conn.execute(
            "INSERT INTO users (email, password_hash, name, role, status, is_verified, created_at) VALUES (?,?,?,?,?,?,?)",
            (FOUNDER_EMAIL, hash_password(FOUNDER_PASSWORD), "Founder", "founder", "active", 1, datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
        logger.info(f"Founder account created for {FOUNDER_EMAIL}")
    conn.close()
    logger.info("Database ready")

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

# ========================= AUTH =========================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: constr(min_length=8)
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/api/register")
def register(data: RegisterRequest):
    email = data.email.lower()
    conn = get_db()
    if conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    role = "founder" if email in FOUNDER_EMAILS else "user"
    conn.execute(
        "INSERT INTO users (email, password_hash, name, role, status, is_verified, created_at) VALUES (?,?,?,?,?,?,?)",
        (email, hash_password(data.password), data.name, role, "active", 1, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    logger.info(f"User registered: {email}")
    return {"success": True, "message": "User registered"}

@app.post("/api/login")
def login(data: LoginRequest):
    email = data.email.lower()
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND status='active'", (email,)).fetchone()
    if not user or not verify_password(data.password, user["password_hash"]):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    conn.close()
    token = create_access_token(email)
    return {
        "access_token": token,
        "user": {"email": user["email"], "name": user["name"], "role": user["role"]}
    }

@app.get("/api/me")
def get_me(current_user=Depends(get_current_user)):
    return {"email": current_user["email"], "name": current_user["name"], "role": current_user["role"]}

# ========================= CHATS =========================
class ChatCreate(BaseModel):
    title: str = "New Chat"

@app.post("/api/chats")
def create_chat(chat: ChatCreate, current_user=Depends(get_current_user)):
    chat_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO chats (id, user_email, title, pinned, created_at) VALUES (?,?,?,?,?)",
        (chat_id, current_user["email"], chat.title, 0, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"id": chat_id, "title": chat.title, "pinned": False, "created_at": datetime.datetime.utcnow().isoformat()}

@app.get("/api/chats")
def get_chats(current_user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, pinned, created_at FROM chats WHERE user_email=? ORDER BY pinned DESC, created_at DESC",
        (current_user["email"],)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, current_user=Depends(get_current_user)):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    msgs = conn.execute("SELECT role, content, created_at FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)).fetchall()
    conn.close()
    return {
        "id": chat["id"],
        "title": chat["title"],
        "pinned": bool(chat["pinned"]),
        "messages": [dict(m) for m in msgs]
    }

@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, update: dict, current_user=Depends(get_current_user)):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    if "title" in update:
        conn.execute("UPDATE chats SET title=? WHERE id=?", (update["title"], chat_id))
    if "pinned" in update:
        conn.execute("UPDATE chats SET pinned=? WHERE id=?", (1 if update["pinned"] else 0, chat_id))
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, current_user=Depends(get_current_user)):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()
    return {"success": True}

# ========================= AI STREAMING =========================
class ChatStreamRequest(BaseModel):
    chat_id: str
    messages: List[dict]
    model: str = "GPT-4o"

MODEL_MAP = {
    "GPT-4o": "openai/gpt-4o",
    "Claude 3.5": "anthropic/claude-sonnet-4-20250514",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash",
    "DeepSeek": "deepseek/deepseek-chat",
    "Llama 3": "meta-llama/llama-3.3-70b-instruct",
    "Mistral Large": "mistralai/mistral-large",
    "Grok-2": "x-ai/grok-2",
    "Perplexity": "perplexity/llama-3.1-sonar-large-128k-online",
    "Cohere": "cohere/command-r-plus",
    "Copilot": "microsoft/copilot",
    "Sonar 2": "perplexity/sonar-reasoning"
}

async def stream_gemini(messages: list):
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'content':'Gemini API key not configured'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    try:
        prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60.0
            )
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Gemini error: {error_text}")
                yield f"data: {json.dumps({'content': f'Gemini Error {response.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            yield f"data: {json.dumps({'content': text})}\n\n"
            yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Gemini exception: {e}")
        yield f"data: {json.dumps({'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

async def stream_openrouter(messages: list, model_display: str, user_email: str):
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'content': 'Error: OpenRouter API key not configured.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    # Inject memory
    conn = get_db()
    memories = conn.execute(
        "SELECT key, value FROM memory WHERE user_email=? AND profile='default'",
        (user_email,)
    ).fetchall()
    conn.close()
    memory_context = " ".join([f"{m['key']}: {m['value']};" for m in memories])
    if memory_context:
        memory_context = "[User context] " + memory_context + "\n"

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = memory_context + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": memory_context})

    model_id = MODEL_MAP.get(model_display, model_display)
    referer = os.getenv("APP_URL", "http://localhost:8000")
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": "AgentOS",
            },
            json={"model": model_id, "messages": messages, "stream": True},
            timeout=60.0
        ) as response:
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"OpenRouter error: {error_text}")
                yield f"data: {json.dumps({'content': f'Error: OpenRouter returned {response.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return
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
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except:
                        pass
            yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
async def stream_chat(req: ChatStreamRequest, current_user=Depends(get_current_user)):
    chat_id = req.chat_id
    model = req.model
    messages = req.messages

    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=403, detail="Not your chat")

    # Save user message
    if messages and messages[-1]["role"] == "user":
        user_msg = messages[-1]["content"]
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?,?,?,?)",
            (chat_id, "user", user_msg, datetime.datetime.utcnow().isoformat())
        )
        # Auto‑title
        if chat["title"] == "New Chat" and user_msg.strip():
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "openai/gpt-3.5-turbo",
                            "messages": [
                                {"role": "system", "content": "Generate a very short title (max 5 words) for a conversation that starts with: " + user_msg}
                            ],
                            "max_tokens": 20,
                            "temperature": 0.3
                        },
                        timeout=10.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        title = data["choices"][0]["message"]["content"].strip().strip('"')
                        if title:
                            conn.execute("UPDATE chats SET title=? WHERE id=?", (title[:60], chat_id))
            except:
                pass
    conn.commit()
    conn.close()

    async def generate():
        accumulated_response = ""
        # Route model
        if model == "Gemini 2.0 Flash" and GEMINI_API_KEY:
            stream = stream_gemini(messages)
        else:
            stream = stream_openrouter(messages, model, current_user["email"])

        async for chunk in stream:
            if chunk.startswith("data: [DONE]"):
                break
            yield chunk
            try:
                payload = json.loads(chunk.replace("data: ", ""))
                accumulated_response += payload.get("content", "")
            except Exception:
                continue

        if accumulated_response:
            conn2 = get_db()
            conn2.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?,?,?,?)",
                (chat_id, "assistant", accumulated_response, datetime.datetime.utcnow().isoformat())
            )
            conn2.commit()
            conn2.close()

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ========================= MEMORY =========================
@app.get("/api/memory")
def get_memory(current_user=Depends(get_current_user), profile: str = "default"):
    conn = get_db()
    items = conn.execute(
        "SELECT key, value FROM memory WHERE user_email=? AND profile=?",
        (current_user["email"], profile)
    ).fetchall()
    conn.close()
    return {"profile": profile, "items": [dict(i) for i in items]}

@app.post("/api/memory")
def add_memory(key: str, value: str, profile: str = "default", current_user=Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO memory (user_email, profile, key, value) VALUES (?,?,?,?)",
        (current_user["email"], profile, key, value)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/memory/{key}")
def delete_memory(key: str, profile: str = "default", current_user=Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "DELETE FROM memory WHERE user_email=? AND profile=? AND key=?",
        (current_user["email"], profile, key)
    )
    conn.commit()
    conn.close()
    return {"success": True}

# ========================= IMAGE & FILE =========================
@app.post("/api/image-analyze")
async def analyze_image(file: UploadFile = File(...), query: str = "Describe this image", current_user=Depends(get_current_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only images allowed")
    content = await file.read()
    if len(content) > 10*1024*1024:
        raise HTTPException(status_code=400, detail="File too large")
    data_url = f"data:{file.content_type};base64,{base64.b64encode(content).decode()}"
    messages = [{"role": "user", "content": [{"type": "text", "text": query}, {"type": "image_url", "image_url": {"url": data_url}}]}]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-4o", "messages": messages, "max_tokens": 300},
            timeout=30.0
        )
        if resp.status_code == 200:
            return {"result": resp.json()["choices"][0]["message"]["content"]}
        return {"result": f"Analysis failed: HTTP {resp.status_code}"}

class ImageGenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024

@app.post("/api/image-generate")
def generate_image(req: ImageGenRequest, current_user=Depends(get_current_user)):
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(req.prompt)}?width={req.width}&height={req.height}&seed={secrets.randbelow(10000)}&nofeed=true"
    return {"url": url}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}

# ========================= ADMIN =========================
@app.get("/api/admin/users", dependencies=[Depends(founder_required)])
def admin_list_users():
    conn = get_db()
    rows = conn.execute("SELECT email, name, role, status, is_verified, created_at FROM users").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.put("/api/admin/users/{email}/ban", dependencies=[Depends(founder_required)])
def admin_ban_user(email: str):
    conn = get_db()
    conn.execute("UPDATE users SET status='banned' WHERE email=?", (email.lower(),))
    conn.commit()
    conn.close()
    return {"success": True}

@app.put("/api/admin/users/{email}/unban", dependencies=[Depends(founder_required)])
def admin_unban_user(email: str):
    conn = get_db()
    conn.execute("UPDATE users SET status='active' WHERE email=?", (email.lower(),))
    conn.commit()
    conn.close()
    return {"success": True}

@app.put("/api/admin/users/{email}/suspend", dependencies=[Depends(founder_required)])
def admin_suspend_user(email: str):
    conn = get_db()
    conn.execute("UPDATE users SET status='suspended' WHERE email=?", (email.lower(),))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/admin/stats", dependencies=[Depends(founder_required)])
def admin_stats():
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    active_users = conn.execute("SELECT COUNT(*) FROM users WHERE status='active'").fetchone()[0]
    total_chats = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
    total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_chats": total_chats,
        "total_messages": total_messages
    }