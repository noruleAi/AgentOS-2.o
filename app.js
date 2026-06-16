/**************************************************************
 *  AgentOS AI v35 – Production Frontend (Final)
 **************************************************************/

const CONFIG = {
  API_BASE: "https://agentos-2o-production-a21a.up.railway.app",
  DB_NAME: "AgentOSDB",
  DB_VERSION: 1,
  FOUNDER_EMAILS: ["rahulkumarmahto2024@gmail.com"],   // compared in lowercase
  ALL_LANGUAGES: [
    "Python","JavaScript","TypeScript","Java","C","C++","C#","Go","Rust","Swift",
    "Kotlin","Dart","PHP","Ruby","Perl","R","MATLAB","Scala","Haskell","Elixir",
    "Erlang","Clojure","OCaml","F#","Groovy","Objective-C","Lua","Julia",
    "Shell (Bash)","PowerShell","SQL","HTML","CSS","Assembly","VHDL","Verilog",
    "Solidity","COBOL","Fortran","Ada","Pascal","Prolog","Lisp","Scheme",
    "Racket","Visual Basic","ABAP","Apex","Crystal","Nim","Zig","Odin","Vala",
    "Haxe","Reason","PureScript"
  ]
};

// ===================== IndexedDB =====================
const Database = (() => {
  let db = null;
  const DB_NAME = CONFIG.DB_NAME;
  const DB_VERSION = CONFIG.DB_VERSION;

  async function open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains("users")) db.createObjectStore("users", { keyPath: "email" });
        if (!db.objectStoreNames.contains("chats")) db.createObjectStore("chats", { keyPath: "id" });
        if (!db.objectStoreNames.contains("settings")) db.createObjectStore("settings", { keyPath: "key" });
      };
      req.onsuccess = (e) => { db = e.target.result; resolve(db); };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async function get(store, key) {
    const tx = db.transaction(store, "readonly");
    const objStore = tx.objectStore(store);
    return new Promise((resolve) => {
      const req = objStore.get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  }

  async function put(store, value) {
    const tx = db.transaction(store, "readwrite");
    const objStore = tx.objectStore(store);
    return new Promise((resolve) => {
      const req = objStore.put(value);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  }

  async function del(store, key) {
    const tx = db.transaction(store, "readwrite");
    const objStore = tx.objectStore(store);
    return new Promise((resolve) => {
      const req = objStore.delete(key);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    });
  }

  async function getAll(store) {
    const tx = db.transaction(store, "readonly");
    const objStore = tx.objectStore(store);
    return new Promise((resolve) => {
      const req = objStore.getAll();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve([]);
    });
  }

  return { open, get, put, del, getAll };
})();

// ===================== Auth =====================
const Auth = (() => {
  let currentUser = null;
  let token = null;

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

  async function register(email, password, name) {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    return await res.json();
  }

  async function login(email, password) {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    const data = await res.json();
    token = data.access_token;
    currentUser = data.user;
    await Database.put("settings", { key: "token", value: token });
    await Database.put("settings", { key: "currentUser", value: currentUser });
    UI.updateAuthUI();
    return currentUser;
  }

  async function logout() {
    token = null;
    currentUser = null;
    await Database.put("settings", { key: "token", value: null });
    await Database.put("settings", { key: "currentUser", value: null });
    UI.updateAuthUI();
  }

  function getToken() { return token; }
  function getUser() { return currentUser; }

  return { restore, register, login, logout, getToken, getUser };
})();

// ===================== Fetch with timeout =====================
async function fetchWithTimeout(url, options = {}, timeout = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  options.signal = controller.signal;
  try {
    const response = await fetch(url, options);
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

// ===================== Chat =====================
const Chat = (() => {
  let currentChatId = null;

  async function create(title = "New Chat") {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
      body: JSON.stringify({ title })
    });
    if (!res.ok) throw new Error("Failed to create chat: " + (await res.text()));
    return await res.json();
  }

  async function loadAll() {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats`, {
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
    if (!res.ok) throw new Error("Failed to load chats");
    return await res.json();
  }

  async function load(chatId) {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
    if (!res.ok) throw new Error("Chat not found");
    return await res.json();
  }

  async function deleteChat(chatId) {
    await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
  }

  async function streamChat(chatId, messages, model = "openai/gpt-3.5-turbo") {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${Auth.getToken()}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ chat_id: chatId, messages, model })
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Stream request failed: ${res.status} ${text}`);
    }
    return res.body;
  }

  return { create, loadAll, load, deleteChat, streamChat, get currentChatId() { return currentChatId; }, set currentChatId(id) { currentChatId = id; } };
})();

// ===================== UI =====================
const UI = (() => {
  function renderMessages(messages) {
    const chatArea = document.getElementById("reply");
    if (!chatArea) return;
    chatArea.innerHTML = "";
    messages.forEach(msg => {
      const div = document.createElement("div");
      div.className = `message ${msg.role === "user" ? "user-msg" : "assistant-msg"}`;
      let content = msg.content || "";
      if (window.DOMPurify) {
        content = DOMPurify.sanitize(content, { ALLOWED_TAGS: ["b","i","strong","em","pre","code","span","br"], ALLOWED_ATTR: ["class"] });
      }
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
    if (!list) return;
    list.innerHTML = "";
    chats.forEach(chat => {
      const div = document.createElement("div");
      div.className = "chat-item";
      div.innerHTML = `<span class="chat-item-title">💬 ${chat.title}</span><button class="chat-item-delete">🗑</button>`;
      div.querySelector(".chat-item-title").addEventListener("click", () => {
        window.loadChat(chat.id);
      });
      div.querySelector(".chat-item-delete").addEventListener("click", (e) => {
        e.stopPropagation();
        window.deleteChat(chat.id);
      });
      list.appendChild(div);
    });
  }

  function updateAuthUI() {
    const user = Auth.getUser();
    const statusEl = document.getElementById("accountStatus");
    if (statusEl) statusEl.textContent = user ? user.name : "Guest";
    const founderPanel = document.getElementById("founderPanel");
    if (founderPanel) {
      if (user && CONFIG.FOUNDER_EMAILS.includes(user.email.toLowerCase())) {
        founderPanel.style.display = "block";
      } else {
        founderPanel.style.display = "none";
      }
    }
  }

  return { renderMessages, renderChatList, updateAuthUI };
})();

// ===================== Global Functions =====================
window.toggleSidebar = function() {
  document.getElementById("sidebar").classList.toggle("active");
};
window.openSettings = function() { window.toggleSidebar(); };
window.newChat = function() {
  if (!Auth.getToken()) return alert("Please login first.");
  const title = prompt("Chat title", "New Chat") || "New Chat";
  Chat.create(title).then(chat => {
    Chat.currentChatId = chat.id;
    Chat.loadAll().then(chats => UI.renderChatList(chats));
    window.loadChat(chat.id);
  }).catch(e => alert(e.message));
};
window.loadChat = async function(chatId) {
  try {
    const chat = await Chat.load(chatId);
    if (chat) {
      Chat.currentChatId = chatId;
      UI.renderMessages(chat.messages);
    }
  } catch (e) {
    alert("Error loading chat: " + e.message);
  }
};

// Safe delete – works both from context menu and list
window.deleteChat = async function(chatId) {
  chatId = chatId || Chat.currentChatId;
  if (!chatId) return;
  if (!confirm("Delete this chat?")) return;
  try {
    await Chat.deleteChat(chatId);
    if (Chat.currentChatId === chatId) {
      Chat.currentChatId = null;
      document.getElementById("reply").innerHTML = '<div class="welcome"><h1>AgentOS AI</h1></div>';
    }
    Chat.loadAll().then(chats => UI.renderChatList(chats));
  } catch (e) {
    alert(e.message);
  }
};

window.toggleMemory = function() {};
window.toggleAutoSave = function() {};
window.toggleTheme = function() {
  document.body.classList.toggle("light");
  const isLight = document.body.classList.contains("light");
  Database.put("settings", { key: "darkTheme", value: !isLight });
  document.getElementById("themeStatus").textContent = isLight ? "Light" : "Dark";
};
window.toggleAutoName = function() {};
window.viewLearningData = function() { alert("Not implemented"); };
window.viewThreatLog = function() { alert("Not implemented"); };

window.showLoginPanel = function() {
  if (Auth.getUser()) {
    // Already logged in – offer logout
    if (confirm(`Logged in as ${Auth.getUser().name}. Logout?`)) {
      Auth.logout();
    }
    return;
  }
  const action = prompt("Type 'login' or 'register' (or cancel):");
  if (!action) return;
  const email = prompt("Email:");
  if (!email) return;
  const password = prompt("Password:");
  if (!password) return;
  if (action === "register") {
    const name = prompt("Your name:");
    Auth.register(email, password, name).then(() => {
      alert("Registered! Now login.");
    }).catch(e => alert(e.message));
  } else if (action === "login") {
    Auth.login(email, password).then(user => {
      alert(`Welcome ${user.name}!`);
      Chat.loadAll().then(chats => UI.renderChatList(chats));
    }).catch(e => alert(e.message));
  }
};

window.toggleVoice = function() {};
window.changeLanguage = function(lang) {};
window.listAllUsers = function() { alert("Admin only"); };
window.banUser = function() { alert("Admin only"); };
window.suspendUser = function() { alert("Admin only"); };
window.unbanUser = function() { alert("Admin only"); };
window.startVoice = function() {};
window.stopGeneration = function() {};
window.openAttachmentMenu = function() {
  document.getElementById("attachmentMenu").classList.toggle("active");
};
window.closeAttachmentMenu = function() {
  document.getElementById("attachmentMenu").classList.remove("active");
};
window.openCamera = function() { document.getElementById("cameraInput").click(); };
window.openGallery = function() { document.getElementById("galleryInput").click(); };
window.openFiles = function() { document.getElementById("fileInput").click(); };
window.openFaceSearch = function() {
  switchMode("facesearch", document.querySelector('[data-mode="facesearch"]'));
  document.getElementById("faceInput").click();
};
window.exportChats = function() { alert("Export coming soon"); };
window.importChats = function() { document.getElementById("importInput").click(); };
window.pinChat = function() { alert("Pin not yet implemented"); };
window.renameChat = function() { alert("Rename not yet implemented"); };

// ===================== Mode & Model Switching =====================
window.switchMode = function(mode, btn) {
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  document.querySelectorAll(".code-mode-ui").forEach(el => el.style.display = "none");
  const input = document.getElementById("msg");
  const placeholderMap = {
    chat: "Message AgentOS AI...",
    code: "Describe code...",
    build: "What to build?",
    hacking: "Ask ethical hacking...",
    agent: "Give a task...",
    bugscanner: "Paste code to scan...",
    webanalyzer: "Paste URL...",
    videoanalyzer: "Upload video...",
    facesearch: "Upload face photo...",
    trading: "Enter stock symbol...",
    math: "Enter math problem...",
    writer: "What to write?",
    legal: "Ask legal question...",
    osint: "Name target...",
    autonomous: "Give complex task..."
  };
  if (input) input.placeholder = placeholderMap[mode] || "Message AgentOS AI...";
  if (mode === "code" || mode === "python" || mode === "webdev") {
    document.getElementById("codeModeUI").style.display = "flex";
  }
  if (mode === "webanalyzer") document.getElementById("webUrlInput").style.display = "flex";
  if (mode === "videoanalyzer") document.getElementById("videoInputUI").style.display = "flex";
  if (mode === "facesearch") document.getElementById("faceSearchUI").style.display = "flex";
  if (mode === "trading") document.getElementById("tradingUI").style.display = "flex";
  if (mode === "build") document.getElementById("buildUI").style.display = "flex";
};

// FIXED: update the internal selectedModel variable
let selectedModel = "GPT-5";
window.selectModel = function(model, btn) {
  selectedModel = model;
  document.querySelectorAll(".model-chip").forEach(b => b.classList.remove("active-model"));
  btn.classList.add("active-model");
  document.getElementById("activeModel").textContent = model;
};

// ===================== Main App =====================
const App = (() => {
  async function init() {
    await Database.open();
    await Auth.restore();
    UI.updateAuthUI();
    if (Auth.getToken()) {
      try {
        const chats = await Chat.loadAll();
        UI.renderChatList(chats);
      } catch (e) { console.error(e); }
    }
    document.getElementById("msg").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  async function sendMessage() {
    if (!Auth.getToken()) return alert("Please login first.");
    const input = document.getElementById("msg");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";

    let chatId = Chat.currentChatId;
    if (!chatId) {
      try {
        const chat = await Chat.create();
        chatId = chat.id;
        Chat.currentChatId = chatId;
        Chat.loadAll().then(chats => UI.renderChatList(chats));
      } catch (e) {
        alert("Error creating chat: " + e.message);
        return;
      }
    }

    const chatArea = document.getElementById("reply");
    const userDiv = document.createElement("div");
    userDiv.className = "message user-msg";
    userDiv.textContent = text;
    chatArea.appendChild(userDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    try {
      const chat = await Chat.load(chatId);
      if (!chat) throw new Error("Chat not found");
      const messages = chat.messages.map(m => ({ role: m.role, content: m.content }));
      messages.push({ role: "user", content: text });

      const modelMap = {
        "GPT-5": "openai/gpt-4o",
        "Claude 3.5": "anthropic/claude-sonnet-4-20250514",
        "Gemini 1.5 Pro": "google/gemini-1.5-pro",
        "DeepSeek": "deepseek/deepseek-chat",
        "Llama 3": "meta-llama/llama-3.3-70b-instruct"
      };
      const modelId = modelMap[selectedModel] || selectedModel;

      const stream = await Chat.streamChat(chatId, messages, modelId);
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      const assistantDiv = document.createElement("div");
      assistantDiv.className = "message assistant-msg";
      chatArea.appendChild(assistantDiv);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                assistantContent += parsed.content;
                // Safe rendering – DOMPurify fallback
                if (window.DOMPurify) {
                  assistantDiv.innerHTML = DOMPurify.sanitize(assistantContent.replace(/\n/g, "<br>"), { ALLOWED_TAGS: ["b","i","strong","em","pre","code","span","br"], ALLOWED_ATTR: ["class"] });
                } else {
                  assistantDiv.textContent = assistantContent;
                }
                chatArea.scrollTop = chatArea.scrollHeight;
              }
            } catch(e) {}
          }
        }
      }

      const updatedChat = await Chat.load(chatId);
      if (updatedChat) UI.renderMessages(updatedChat.messages);
    } catch (e) {
      alert("Error: " + e.message);
    }
  }

  return { init, sendMessage };
})();

window.sendMessage = () => App.sendMessage();

window.addEventListener("load", () => {
  App.init().catch(e => {
    console.error(e);
    alert("Initialization error: " + e.message);
  });
});