/**************************************************************
 *  AgentOS AI v35 – Production Frontend
 *  IndexedDB, JWT, DOMPurify, Offline Detection, Modular
 **************************************************************/

// ===================== 1. CONFIG =====================
const CONFIG = {
  API_BASE: "https://agentos-2o-production.up.railway.app",
  DB_NAME: "AgentOSDB",
  DB_VERSION: 1,
  FOUNDER_EMAILS: ["rahulkumarmahto2024@gmail.com"],
  ALL_LANGUAGES: [ /* 55+ languages */ ]
};

// ===================== 2. DATABASE (IndexedDB) =====================
const Database = (() => {
  let db = null;

  async function open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(CONFIG.DB_NAME, CONFIG.DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains("users")) db.createObjectStore("users", { keyPath: "email" });
        if (!db.objectStoreNames.contains("chats")) db.createObjectStore("chats", { keyPath: "id" });
        if (!db.objectStoreNames.contains("settings")) db.createObjectStore("settings", { keyPath: "key" });
        // Index for chat search
        if (!db.objectStoreNames.contains("messages")) {
          const msgStore = db.createObjectStore("messages", { keyPath: "id", autoIncrement: true });
          msgStore.createIndex("chatId", "chatId", { unique: false });
        }
      };
      req.onsuccess = (e) => { db = e.target.result; resolve(db); };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async function get(store, key) { /* ... */ }
  async function put(store, value) { /* ... */ }
  async function del(store, key) { /* ... */ }
  async function getAll(store) { /* ... */ }

  return { open, get, put, del, getAll };
})();

// ===================== 3. AUTH =====================
const Auth = (() => {
  let currentUser = null;
  let token = null;

  async function hashPassword(password, salt) {
    const encoder = new TextEncoder();
    const data = encoder.encode(password + (salt || ""));
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashHex = Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, "0")).join("");
    if (!salt) {
      salt = Array.from(crypto.getRandomValues(new Uint8Array(16))).map(b => b.toString(16).padStart(2, "0")).join("");
    }
    return { hash: hashHex, salt };
  }

  async function register(email, password, name) {
    // Use backend for registration
    const res = await fetch(`${CONFIG.API_BASE}/api/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    return await res.json();
  }

  async function login(email, password) {
    const res = await fetch(`${CONFIG.API_BASE}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    const data = await res.json();
    token = data.access_token;
    currentUser = data.user;
    // Store token and user in IndexedDB settings
    await Database.put("settings", { key: "token", value: token });
    await Database.put("settings", { key: "currentUser", value: currentUser });
    return currentUser;
  }

  async function logout() {
    token = null;
    currentUser = null;
    await Database.put("settings", { key: "token", value: null });
    await Database.put("settings", { key: "currentUser", value: null });
  }

  function getToken() { return token; }
  function getUser() { return currentUser; }

  // Auto-restore from DB
  async function restore() {
    const storedToken = await Database.get("settings", "token");
    if (storedToken && storedToken.value) {
      token = storedToken.value;
      const storedUser = await Database.get("settings", "currentUser");
      if (storedUser && storedUser.value) {
        currentUser = storedUser.value;
      }
    }
  }

  return { register, login, logout, getToken, getUser, restore };
})();

// ===================== 4. CHAT =====================
const Chat = (() => {
  let currentChatId = null;

  async function create(title = "New Chat") {
    const res = await fetch(`${CONFIG.API_BASE}/api/chats`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
      body: JSON.stringify({ title })
    });
    if (!res.ok) throw new Error("Failed to create chat");
    return await res.json();
  }

  async function loadAll() {
    const res = await fetch(`${CONFIG.API_BASE}/api/chats`, {
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
    return res.ok ? await res.json() : [];
  }

  async function load(chatId) {
    const res = await fetch(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
    return res.ok ? await res.json() : null;
  }

  async function sendMessage(chatId, text) {

    const res = await fetch(
        `${CONFIG.API_BASE}/api/chats/${chatId}/message`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                role: "user",
                content: text
            })
        }
    );

    return await res.json();
}

  async function deleteChat(chatId) {
    await fetch(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
  }

  return { create, loadAll, load, sendMessage, deleteChat };
})();

// ===================== 5. UI =====================
const UI = (() => {
  function renderMessages(messages) {
    const chatArea = document.getElementById("chatArea");
    chatArea.innerHTML = "";
    messages.forEach(msg => {
      const div = document.createElement("div");
      div.className = `message ${msg.role === "user" ? "user-msg" : "assistant-msg"}`;
      // DOMPurify sanitizes and allows only safe HTML
      let content = DOMPurify.sanitize(msg.content, { ALLOWED_TAGS: ["b","i","strong","em","pre","code","span"], ALLOWED_ATTR: ["class"] });
      // Convert code blocks manually for syntax highlighting
      content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre class="code-block"><code>${code.replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code></pre>`;
      });
      div.innerHTML = content;
      chatArea.appendChild(div);
    });
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function renderChatList(chats) {
    const list = document.getElementById("chatList");
    list.innerHTML = "";
    chats.forEach(chat => {
      const div = document.createElement("div");
      div.className = "chat-item";
      div.innerHTML = `<span class="chat-item-title">💬 ${chat.title}</span><button class="chat-item-delete">🗑</button>`;
      div.querySelector(".chat-item-title").onclick = () => loadChat(chat.id);
      div.querySelector(".chat-item-delete").onclick = (e) => { e.stopPropagation(); deleteChat(chat.id); };
      list.appendChild(div);
    });
  }

  // ... other UI helpers (toggleSidebar, switchMode, etc.)

  return { renderMessages, renderChatList };
})();

// ===================== 6. MAIN APP =====================
const App = (() => {
  let currentMode = "chat";
  let selectedModel = "openai/gpt-3.5-turbo";
  let speechActive = true;
  let darkTheme = true;

  async function init() {
    await Database.open();
    await Auth.restore();
    updateAuthUI();

    // Load chats if logged in
    if (Auth.getUser()) {
      const chats = await Chat.loadAll();
      UI.renderChatList(chats);
      if (chats.length) {
        const chat = await Chat.load(chats[0].id);
        if (chat) UI.renderMessages(chat.messages);
      }
    }

    // Setup event listeners
    document.getElementById("msg").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    // ... other listeners
  }

  async function sendMessage() {
    if (!Auth.getUser()) return alert("Please login");
    const input = document.getElementById("msg");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";

    let chatId = currentChatId;
    if (!chatId) {
      const chat = await Chat.create();
      chatId = chat.id;
      currentChatId = chatId;
      // Refresh chat list
      const chats = await Chat.loadAll();
      UI.renderChatList(chats);
    }

    // Add user message locally (optimistic)
    const userMsg = { role: "user", content: text };
    // UI.renderMessages will be called after stream

    // Build messages array from loaded chat
    const chat = await Chat.load(chatId);
    const messages = chat.messages.map(m => ({ role: m.role, content: m.content }));
    messages.push({ role: "user", content: text });

    const result = await Chat.sendMessage(chatId, text);

console.log(result);

if (result.messages) {
    UI.renderMessages(result.messages);
}

} // <-- CLOSE sendMessage() HERE

// Theme toggle
function toggleTheme() {
    darkTheme = !darkTheme;
    document.body.classList.toggle("light", !darkTheme);
    Database.put("settings", {
        key: "darkTheme",
        value: darkTheme
    });
}

  // ... other functions (switchMode, selectModel, etc.)

  return { init, sendMessage, toggleTheme };
})();

window.sendMessage = () => App.sendMessage();

console.log("APP LOADED");

window.addEventListener("load", () => App.init());