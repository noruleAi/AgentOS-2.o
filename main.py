import os, json, uuid, datetime, logging, secrets, base64, hashlib, sqlite3, urllib.parse
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, constr, validator
from jose import jwt, JWTError
import httpx
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ========================= CONFIGURATION =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT", "15"))
MAX_MESSAGE_LENGTH = 10000
MAX_RESPONSE_CHARS = 500000

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://agentos-ai.in").split(",")
FOUNDER_EMAILS = set(email.strip().lower() for email in os.getenv("FOUNDER_EMAILS", "").split(",") if email.strip())
FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "rahulkumarmahto2024@gmail.com").lower()
FOUNDER_PASSWORD = os.getenv("FOUNDER_PASSWORD")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentos")

# ========================= Password Hashing =========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)
def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)

# ========================= SQLite Database =========================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentos.db")
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY, password_hash TEXT NOT NULL, name TEXT NOT NULL,
            role TEXT DEFAULT 'user', status TEXT DEFAULT 'active', is_verified INTEGER DEFAULT 0,
            avatar_url TEXT DEFAULT '', verification_code TEXT DEFAULT '', push_sub TEXT DEFAULT '', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL,
            token TEXT NOT NULL UNIQUE, expires_at TEXT NOT NULL, used INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, user_email TEXT NOT NULL, title TEXT NOT NULL DEFAULT 'New Chat',
            pinned INTEGER DEFAULT 0, created_at TEXT,
            FOREIGN KEY(user_email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'default', key TEXT NOT NULL, value TEXT NOT NULL,
            UNIQUE(user_email, profile, key),
            FOREIGN KEY(user_email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS settings (
            user_email TEXT PRIMARY KEY, theme TEXT DEFAULT 'dark', language TEXT DEFAULT 'en',
            speech_active INTEGER DEFAULT 1, memory_enabled INTEGER DEFAULT 1,
            auto_save INTEGER DEFAULT 1, auto_name INTEGER DEFAULT 1,
            FOREIGN KEY(user_email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT (datetime('now')),
            admin_email TEXT, action TEXT, target TEXT, details TEXT
        );
        -- Performance indexes
        CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_email);
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_memory_user ON memory(user_email);
    """)
    conn.execute("DELETE FROM password_resets WHERE expires_at < datetime('now')")
    conn.commit()
    conn.close()

# ========================= FastAPI App =========================
app = FastAPI(title="AgentOS AI 2.0", version="2.0")

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = BASE_DIR
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if os.path.exists(os.path.join(BASE_DIR, "images")):
    app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "images")), name="images")

security = HTTPBearer()

def create_access_token(email: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status_code=401)
    email = payload.get("sub")
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND status='active'", (email,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

def founder_required(user=Depends(get_current_user)):
    if user.get("role") != "founder":
        raise HTTPException(status_code=403)
    return user

def log_admin_action(admin_email: str, action: str, target: str, details: str = ""):
    try:
        conn = get_db()
        conn.execute("INSERT INTO admin_logs (admin_email, action, target, details) VALUES (?,?,?,?)",
                     (admin_email, action, target, details))
        conn.commit()
        conn.close()
    except:
        pass

@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal error"})

@app.on_event("startup")
async def startup():
    init_db()
    if FOUNDER_PASSWORD:
        try:
            conn = get_db()
            if not conn.execute("SELECT 1 FROM users WHERE email=?", (FOUNDER_EMAIL,)).fetchone():
                conn.execute("INSERT INTO users (email, password_hash, name, role, status, is_verified, created_at) VALUES (?,?,?,?,?,?,?)",
                             (FOUNDER_EMAIL, hash_password(FOUNDER_PASSWORD), "Founder", "founder", "active", 1, datetime.datetime.utcnow().isoformat()))
                conn.commit()
            conn.close()
        except: pass
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
async def register(data: RegisterRequest):
    email = data.email.lower()
    try:
        conn = get_db()
        if conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Email already registered")
        verification_code = str(secrets.randbelow(900000) + 100000)
        role = "founder" if email in FOUNDER_EMAILS else "user"
        conn.execute("INSERT INTO users (email, password_hash, name, role, status, is_verified, verification_code, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (email, hash_password(data.password), data.name, role, "active", 0, verification_code, datetime.datetime.utcnow().isoformat()))
        conn.execute("INSERT OR IGNORE INTO settings (user_email) VALUES (?)", (email,))
        conn.commit()
        conn.close()
        # In production, send email with verification code. We'll just log it for demo.
        logger.info(f"Verification code for {email}: {verification_code}")
        return {"success": True, "message": "User registered. Check your email for verification code."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500)

@app.post("/api/verify-email")
async def verify_email(email: EmailStr, code: str):
    try:
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND verification_code=? AND is_verified=0", (email, code)).fetchone()
        if not user:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid or expired code")
        conn.execute("UPDATE users SET is_verified=1, verification_code=NULL WHERE email=?", (email,))
        conn.commit()
        conn.close()
        return {"message": "Email verified successfully"}
    except HTTPException:
        raise
    except:
        raise HTTPException(status_code=500)

@app.post("/api/login")
async def login(data: LoginRequest):
    email = data.email.lower()
    try:
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND status='active'", (email,)).fetchone()
        if not user or not verify_password(data.password, user["password_hash"]):
            conn.close()
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not user["is_verified"]:
            conn.close()
            raise HTTPException(status_code=403, detail="Email not verified")
        conn.close()
        token = create_access_token(email)
        return {"access_token": token, "user": {"email": user["email"], "name": user["name"], "role": user["role"], "avatar_url": user["avatar_url"]}}
    finally:
        try: conn.close()
        except: pass

@app.get("/api/me")
def get_me(current_user=Depends(get_current_user)):
    return {"email": current_user["email"], "name": current_user["name"], "role": current_user["role"], "avatar_url": current_user["avatar_url"]}

# Forgot/Reset Password
@app.post("/api/forgot-password")
async def forgot_password(email: EmailStr):
    try:
        conn = get_db()
        user = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
        if not user:
            return {"message": "If the email exists, a reset link has been sent."}
        token = secrets.token_urlsafe(32)
        expires = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        conn.execute("INSERT INTO password_resets (email, token, expires_at) VALUES (?,?,?)", (email, token, expires.isoformat()))
        conn.commit()
        conn.close()
        reset_link = f"{os.getenv('APP_URL','http://localhost:8000')}/reset-password?token={token}"
        logger.info(f"Reset link for {email}: {reset_link}")
        return {"message": "If the email exists, a reset link has been sent."}
    finally:
        try: conn.close()
        except: pass

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: constr(min_length=8)

@app.post("/api/reset-password")
async def reset_password(data: ResetPasswordRequest):
    try:
        conn = get_db()
        row = conn.execute("SELECT email, used FROM password_resets WHERE token=? AND expires_at > ? AND used=0",
                           (data.token, datetime.datetime.utcnow().isoformat())).fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        new_hash = hash_password(data.new_password)
        conn.execute("UPDATE users SET password_hash=? WHERE email=?", (new_hash, row["email"]))
        conn.execute("UPDATE password_resets SET used=1 WHERE token=?", (data.token,))
        conn.commit()
        conn.close()
        return {"message": "Password updated"}
    finally:
        try: conn.close()
        except: pass

# ========================= CHATS =========================
class ChatCreate(BaseModel):
    title: str = "New Chat"
class ChatUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
class ChatMessage(BaseModel):
    role: str
    content: str
    @validator('content')
    def max_len(cls, v):
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError('Too long')
        return v

@app.post("/api/chats")
async def create_chat(chat: ChatCreate, current_user=Depends(get_current_user)):
    chat_id = str(uuid.uuid4())
    try:
        conn = get_db()
        conn.execute("INSERT INTO chats (id, user_email, title, pinned, created_at) VALUES (?,?,?,?,?)",
                     (chat_id, current_user["email"], chat.title, 0, datetime.datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return {"id": chat_id, "title": chat.title, "pinned": False, "created_at": datetime.datetime.utcnow().isoformat()}
    finally:
        try: conn.close()
        except: pass

@app.get("/api/chats")
def get_chats(current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        rows = conn.execute("SELECT id, title, pinned, created_at FROM chats WHERE user_email=? ORDER BY pinned DESC, created_at DESC",
                            (current_user["email"],)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    finally:
        try: conn.close()
        except: pass

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
        if not chat:
            conn.close()
            raise HTTPException(status_code=404)
        msgs = conn.execute("SELECT role, content, created_at FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)).fetchall()
        conn.close()
        return {"id": chat["id"], "title": chat["title"], "pinned": bool(chat["pinned"]), "messages": [dict(m) for m in msgs]}
    finally:
        try: conn.close()
        except: pass

@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, update: ChatUpdate, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
        if not chat:
            conn.close()
            raise HTTPException(status_code=404)
        if update.title is not None:
            conn.execute("UPDATE chats SET title=? WHERE id=?", (update.title, chat_id))
        if update.pinned is not None:
            conn.execute("UPDATE chats SET pinned=? WHERE id=?", (1 if update.pinned else 0, chat_id))
        conn.commit()
        conn.close()
        return {"success": True}
    finally:
        try: conn.close()
        except: pass

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"]))
        conn.commit()
        conn.close()
        return {"success": True}
    finally:
        try: conn.close()
        except: pass

@app.post("/api/chats/{chat_id}/messages")
def add_message(chat_id: str, msg: ChatMessage, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        conn.execute("INSERT INTO messages (chat_id, role, content, created_at) VALUES (?,?,?,?)",
                     (chat_id, msg.role, msg.content, datetime.datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return {"success": True}
    finally:
        try: conn.close()
        except: pass

@app.get("/api/chats/{chat_id}/export")
def export_chat(chat_id: str, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
        if not chat:
            raise HTTPException(status_code=404)
        msgs = conn.execute("SELECT role, content, created_at FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)).fetchall()
        conn.close()
        return {"chat": {"id": chat["id"], "title": chat["title"], "pinned": bool(chat["pinned"])}, "messages": [dict(m) for m in msgs]}
    finally:
        try: conn.close()
        except: pass

# ========================= AI STREAMING =========================
class ChatStreamRequest(BaseModel):
    chat_id: str
    messages: List[dict]
    model: str = "GPT-4o"
    tool: Optional[str] = "No Tool"
    mode: Optional[str] = "chat"

MODEL_MAP = {
    "GPT-4o": "openai/gpt-4o", "Claude 3.5": "anthropic/claude-sonnet-4-20250514",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash", "DeepSeek": "deepseek/deepseek-chat",
    "Llama 3": "meta-llama/llama-3.3-70b-instruct", "Grok-2": "x-ai/grok-2",
    "Perplexity": "perplexity/llama-3.1-sonar-large-128k-online", "Cohere": "cohere/command-r-plus",
    "Bedrock": "anthropic/claude-sonnet-4-20250514", "Mistral Large": "mistralai/mistral-large",
    "Copilot": "microsoft/copilot", "Sonar 2": "perplexity/sonar-reasoning"
}

async def stream_gemini(messages):
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'content':'Gemini key missing'})}\n\n"
        yield "data: [DONE]\n\n"; return
    prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                                 json={"contents":[{"parts":[{"text":prompt}]}]}, timeout=180)
        if resp.status_code != 200:
            yield f"data: {json.dumps({'content':f'Gemini error {resp.status_code}'})}\n\n"
            yield "data: [DONE]\n\n"; return
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        yield f"data: {json.dumps({'content': text})}\n\n"
        yield "data: [DONE]\n\n"

async def stream_openrouter(messages, model_display, user_email):
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'content':'OpenRouter key missing'})}\n\n"
        yield "data: [DONE]\n\n"; return
    try:
        conn = get_db()
        memories = conn.execute("SELECT key, value FROM memory WHERE user_email=? AND profile='default'", (user_email,)).fetchall()
        conn.close()
        mem_text = " ".join([f"{m['key']}:{m['value']};" for m in memories])
    except:
        mem_text = ""
    if mem_text:
        mem_text = "[User context] " + mem_text + "\n"
    if messages and messages[0]["role"]=="system":
        messages[0]["content"] = mem_text + messages[0]["content"]
    else:
        messages.insert(0, {"role":"system","content":mem_text})
    model_id = MODEL_MAP.get(model_display, model_display)
    referer = os.getenv("APP_URL", "http://localhost:8000")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":referer},
            json={"model":model_id,"messages":messages,"stream":True}, timeout=180) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'content':f'OpenRouter error {resp.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"; return
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    d = line[6:].strip()
                    if d == "[DONE]": break
                    try:
                        chunk = json.loads(d)
                        content = chunk["choices"][0]["delta"].get("content","")
                        if content: yield f"data: {json.dumps({'content': content})}\n\n"
                    except: pass
            yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def stream_chat(request: Request, req: ChatStreamRequest, current_user=Depends(get_current_user)):
    chat_id, model, messages = req.chat_id, req.model, req.messages
    try:
        conn = get_db()
        chat = conn.execute("SELECT * FROM chats WHERE id=? AND user_email=?", (chat_id, current_user["email"])).fetchone()
        if not chat:
            conn.close()
            raise HTTPException(status_code=403)
        if messages and messages[-1]["role"]=="user":
            user_msg = messages[-1]["content"]
            conn.execute("INSERT INTO messages (chat_id,role,content,created_at) VALUES (?,?,?,?)",
                         (chat_id,"user",user_msg,datetime.datetime.utcnow().isoformat()))
            if chat["title"]=="New Chat" and user_msg.strip():
                try:
                    async with httpx.AsyncClient() as c:
                        r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"},
                            json={"model":"openai/gpt-4o-mini","messages":[{"role":"system","content":"Generate a very short title (max 5 words) for a conversation that starts with: "+user_msg}],"max_tokens":20,"temperature":0.3},
                            timeout=10)
                        if r.status_code==200:
                            title = r.json()["choices"][0]["message"]["content"].strip().strip('"')
                            if title: conn.execute("UPDATE chats SET title=? WHERE id=?", (title[:60], chat_id))
                except: pass
        conn.commit()
        conn.close()
    finally:
        try: conn.close()
        except: pass

    async def generate():
        acc = ""
        s = stream_gemini(messages) if model=="Gemini 2.0 Flash" and GEMINI_API_KEY else stream_openrouter(messages, model, current_user["email"])
        async for chunk in s:
            if chunk.startswith("data: [DONE]"): break
            yield chunk
            try:
                payload = json.loads(chunk.replace("data: ", ""))
                acc += payload.get("content","")
                if len(acc) > MAX_RESPONSE_CHARS:
                    acc = acc[-MAX_RESPONSE_CHARS:]
            except: pass
        if acc:
            try:
                conn2 = get_db()
                conn2.execute("INSERT INTO messages (chat_id,role,content,created_at) VALUES (?,?,?,?)",
                              (chat_id,"assistant",acc[:MAX_RESPONSE_CHARS],datetime.datetime.utcnow().isoformat()))
                conn2.commit()
                conn2.close()
            except: pass
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

# ========================= MEMORY / SETTINGS / IMAGE / SEARCH / VOICE =========================
@app.get("/api/memory")
def get_memory(current_user=Depends(get_current_user), profile: str="default"):
    try:
        conn = get_db()
        items = conn.execute("SELECT key, value FROM memory WHERE user_email=? AND profile=?", (current_user["email"], profile)).fetchall()
        conn.close()
        return {"profile":profile,"items":[dict(i) for i in items]}
    finally:
        try: conn.close()
        except: pass

@app.post("/api/memory")
def add_memory(key:str, value:str, profile:str="default", current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        conn.execute("INSERT OR REPLACE INTO memory (user_email,profile,key,value) VALUES (?,?,?,?)", (current_user["email"],profile,key,value))
        conn.commit()
        conn.close()
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.delete("/api/memory/{key}")
def delete_memory(key:str, profile:str="default", current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        conn.execute("DELETE FROM memory WHERE user_email=? AND profile=? AND key=?", (current_user["email"],profile,key))
        conn.commit()
        conn.close()
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

# FIXED settings endpoint: safe mapping
SETTING_COLUMNS = {
    "theme": "theme", "language": "language", "speech_active": "speech_active",
    "memory_enabled": "memory_enabled", "auto_save": "auto_save", "auto_name": "auto_name"
}
@app.get("/api/settings")
def get_settings(current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM settings WHERE user_email=?", (current_user["email"],)).fetchone()
        if not row:
            conn.execute("INSERT INTO settings (user_email) VALUES (?)", (current_user["email"],))
            conn.commit()
            row = conn.execute("SELECT * FROM settings WHERE user_email=?", (current_user["email"],)).fetchone()
        conn.close()
        return dict(row)
    finally:
        try: conn.close()
        except: pass

@app.put("/api/settings")
def update_settings(update: dict, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        conn.execute("INSERT OR IGNORE INTO settings (user_email) VALUES (?)", (current_user["email"],))
        for key, value in update.items():
            column = SETTING_COLUMNS.get(key)
            if column:
                conn.execute(f"UPDATE settings SET {column}=? WHERE user_email=?", (value, current_user["email"]))
        conn.commit()
        conn.close()
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

# Image generation/analysis, search, voice, courses unchanged
class ImageGenRequest(BaseModel):
    prompt: str; width:int=1024; height:int=1024
@app.post("/api/image-generate")
def generate_image(req:ImageGenRequest):
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(req.prompt)}?width={req.width}&height={req.height}&seed={secrets.randbelow(10000)}&nofeed=true"
    return {"url":url}

@app.post("/api/image-analyze")
async def analyze_image(file:UploadFile=File(...), query:str="Describe this image", current_user=Depends(get_current_user)):
    content = await file.read()
    if len(content)>10*1024*1024: raise HTTPException(400,"Too large")
    data_url = f"data:{file.content_type};base64,{base64.b64encode(content).decode()}"
    messages = [{"role":"user","content":[{"type":"text","text":query},{"type":"image_url","image_url":{"url":data_url}}]}]
    async with httpx.AsyncClient() as c:
        r = await c.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"},
                          json={"model":"openai/gpt-4o","messages":messages,"max_tokens":300}, timeout=30)
        if r.status_code==200: return {"result":r.json()["choices"][0]["message"]["content"]}
        return {"result":f"Analysis failed: HTTP {r.status_code}"}

@app.get("/api/search")
async def search_web(q:str):
    if TAVILY_API_KEY:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post("https://api.tavily.com/search", json={"api_key":TAVILY_API_KEY,"query":q,"search_depth":"basic","max_results":5}, timeout=15)
                if r.status_code==200:
                    data = r.json()
                    return {"results":[{"title":i["title"],"url":i["url"],"snippet":i["content"]} for i in data.get("results",[])]}
        except Exception as e: logger.error(e)
    return {"results":[{"title":"Search not configured","url":"","snippet":"Set TAVILY_API_KEY"}]}

@app.post("/api/voice")
async def voice_interaction(query:str):
    async with httpx.AsyncClient() as c:
        r = await c.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"},
                          json={"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":query}],"max_tokens":150}, timeout=15)
        if r.status_code==200: return {"reply":r.json()["choices"][0]["message"]["content"]}
        return {"reply":f"Error {r.status_code}"}

@app.get("/api/cybersecurity/courses")
def get_courses():
    return [
        {"level":"Beginner","title":"Google Cybersecurity Certificate","provider":"Google/Coursera","url":"https://www.coursera.org/professional-certificates/google-cybersecurity"},
        {"level":"Beginner","title":"ISC2 CC","provider":"ISC2","url":"https://www.isc2.org/certifications/cc"},
        {"level":"Beginner","title":"CompTIA Security+","provider":"CompTIA","url":"https://www.comptia.org/certifications/security"},
        {"level":"Intermediate","title":"CEH","provider":"EC-Council","url":"https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/"},
        {"level":"Intermediate","title":"CySA+","provider":"CompTIA","url":"https://www.comptia.org/certifications/cybersecurity-analyst"},
        {"level":"Advanced","title":"OSCP","provider":"OffSec","url":"https://www.offsec.com/courses/pen-200/"},
        {"level":"Advanced","title":"CISSP","provider":"ISC2","url":"https://www.isc2.org/certifications/cissp"}
    ]

# ========================= ADMIN =========================
@app.get("/api/admin/users", dependencies=[Depends(founder_required)])
def admin_list_users():
    try:
        conn = get_db()
        rows = conn.execute("SELECT email, name, role, status, is_verified, created_at FROM users").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    finally:
        try: conn.close()
        except: pass

@app.put("/api/admin/users/{email}/ban", dependencies=[Depends(founder_required)])
def admin_ban(email: str, admin=Depends(founder_required)):
    try:
        conn = get_db()
        conn.execute("UPDATE users SET status='banned' WHERE email=?", (email.lower(),))
        conn.commit()
        conn.close()
        log_admin_action(admin["email"], "ban", email)
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.put("/api/admin/users/{email}/unban", dependencies=[Depends(founder_required)])
def admin_unban(email: str, admin=Depends(founder_required)):
    try:
        conn = get_db()
        conn.execute("UPDATE users SET status='active' WHERE email=?", (email.lower(),))
        conn.commit()
        conn.close()
        log_admin_action(admin["email"], "unban", email)
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.put("/api/admin/users/{email}/suspend", dependencies=[Depends(founder_required)])
def admin_suspend(email: str, admin=Depends(founder_required)):
    try:
        conn = get_db()
        conn.execute("UPDATE users SET status='suspended' WHERE email=?", (email.lower(),))
        conn.commit()
        conn.close()
        log_admin_action(admin["email"], "suspend", email)
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.delete("/api/admin/users/{email}", dependencies=[Depends(founder_required)])
def admin_delete_user(email: str, admin=Depends(founder_required)):
    try:
        conn = get_db()
        conn.execute("DELETE FROM users WHERE email=?", (email.lower(),))
        conn.commit()
        conn.close()
        log_admin_action(admin["email"], "delete", email)
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.get("/api/admin/chats", dependencies=[Depends(founder_required)])
def admin_list_chats():
    try:
        conn = get_db()
        rows = conn.execute("SELECT id, user_email, title, pinned, created_at FROM chats ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    finally:
        try: conn.close()
        except: pass

@app.get("/api/admin/chats/{chat_id}", dependencies=[Depends(founder_required)])
def admin_view_chat(chat_id: str):
    try:
        conn = get_db()
        chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not chat: raise HTTPException(status_code=404)
        msgs = conn.execute("SELECT role, content, created_at FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)).fetchall()
        conn.close()
        return {"chat":dict(chat), "messages":[dict(m) for m in msgs]}
    finally:
        try: conn.close()
        except: pass

@app.delete("/api/admin/chats/{chat_id}", dependencies=[Depends(founder_required)])
def admin_delete_chat(chat_id: str, admin=Depends(founder_required)):
    try:
        conn = get_db()
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        conn.commit()
        conn.close()
        log_admin_action(admin["email"], "delete_chat", chat_id)
        return {"success":True}
    finally:
        try: conn.close()
        except: pass

@app.get("/api/admin/logs", dependencies=[Depends(founder_required)])
def admin_logs(limit: int=50):
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM admin_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    finally:
        try: conn.close()
        except: pass

@app.get("/api/admin/stats", dependencies=[Depends(founder_required)])
def admin_stats():
    try:
        conn = get_db()
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(*) FROM users WHERE status='active'").fetchone()[0]
        total_chats = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        return {"total_users":total_users,"active_users":active_users,"total_chats":total_chats,"total_messages":total_messages}
    finally:
        try: conn.close()
        except: pass

# ========================= PROFILE =========================
class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None

@app.put("/api/profile")
async def update_profile(data: ProfileUpdate, current_user=Depends(get_current_user)):
    try:
        conn = get_db()
        if data.name is not None:
            conn.execute("UPDATE users SET name=? WHERE email=?", (data.name, current_user["email"]))
        if data.avatar_url is not None:
            conn.execute("UPDATE users SET avatar_url=? WHERE email=?", (data.avatar_url, current_user["email"]))
        conn.commit()
        conn.close()
        return {"success":True}
    finally:
        try: conn.close()
        except: pass