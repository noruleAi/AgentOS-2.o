/**************************************************************
 *  AgentOS AI v1 Beta – Production Frontend (Final)
 **************************************************************/

const CONFIG = {
  API_BASE: window.location.origin,
  DB_NAME: "AgentOSDB",
  DB_VERSION: 1,
  FOUNDER_EMAILS: ["rahulkumarmahto2024@gmail.com"],
  ALL_LANGUAGES: ["Python","JavaScript","TypeScript","Java","C","C++","C#","Go","Rust","Swift","Kotlin","Dart","PHP","Ruby","Perl","R","MATLAB","Scala","Haskell","Elixir","Erlang","Clojure","OCaml","F#","Groovy","Objective-C","Lua","Julia","Shell (Bash)","PowerShell","SQL","HTML","CSS","Assembly","VHDL","Verilog","Solidity","COBOL","Fortran","Ada","Pascal","Prolog","Lisp","Scheme","Racket","Visual Basic","ABAP","Apex","Crystal","Nim","Zig","Odin","Vala","Haxe","Reason","PureScript"]
};

// ===================== IndexedDB =====================
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
      };
      req.onsuccess = (e) => { db = e.target.result; resolve(db); };
      req.onerror = (e) => reject(e.target.error);
    });
  }
  async function get(store, key) {
    const tx = db.transaction(store, "readonly");
    const obj = tx.objectStore(store);
    return new Promise(r => { const req = obj.get(key); req.onsuccess = () => r(req.result); req.onerror = () => r(null); });
  }
  async function put(store, value) {
    const tx = db.transaction(store, "readwrite");
    const obj = tx.objectStore(store);
    return new Promise(r => { const req = obj.put(value); req.onsuccess = () => r(req.result); req.onerror = () => r(null); });
  }
  async function del(store, key) {
    const tx = db.transaction(store, "readwrite");
    const obj = tx.objectStore(store);
    return new Promise(r => { const req = obj.delete(key); req.onsuccess = () => r(); req.onerror = () => r(); });
  }
  async function getAll(store) {
    const tx = db.transaction(store, "readonly");
    const obj = tx.objectStore(store);
    return new Promise(r => { const req = obj.getAll(); req.onsuccess = () => r(req.result); req.onerror = () => r([]); });
  }
  return { open, get, put, del, getAll };
})();

// ===================== Auth =====================
const Auth = (() => {
  let currentUser = null, token = null;

  async function restore() {
    const storedToken = await Database.get("settings", "token");
    token = storedToken?.value || localStorage.getItem("agentos_token");
    if (token) {
      try {
        const res = await fetch(`${CONFIG.API_BASE}/api/me`, { headers: { Authorization: `Bearer ${token}` } });
        if (res.ok) {
          currentUser = await res.json();
          await Database.put("settings", { key: "token", value: token });
          await Database.put("settings", { key: "currentUser", value: currentUser });
          localStorage.setItem("agentos_token", token);
        } else {
          token = null; currentUser = null;
          localStorage.removeItem("agentos_token");
        }
      } catch { token = null; currentUser = null; }
    }
  }

  async function login(email, password) {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/login`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    const data = await res.json();
    token = data.access_token; currentUser = data.user;
    await Database.put("settings", { key: "token", value: token });
    await Database.put("settings", { key: "currentUser", value: currentUser });
    localStorage.setItem("agentos_token", token);
    UI.updateAuthUI();
    Profile.load();
    return currentUser;
  }

  async function register(email, password, name) {
    const res = await fetchWithTimeout(`${CONFIG.API_BASE}/api/register`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    return await res.json();
  }

  async function logout() {
    token = null; currentUser = null;
    await Database.put("settings", { key: "token", value: null });
    await Database.put("settings", { key: "currentUser", value: null });
    localStorage.removeItem("agentos_token");
    UI.updateAuthUI();
    Chat.loadAll().then(chats => UI.renderChatList(chats));
  }

  function getToken() { return token; }
  function getUser() { return currentUser; }

  function showLoginModal() { document.getElementById("loginModal").classList.remove("hidden"); }
  function closeLoginModal() { document.getElementById("loginModal").classList.add("hidden"); }
  async function doLogin() {
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;
    if (!email || !password) return showToast("Please fill email and password");
    try {
      await login(email, password);
      closeLoginModal();
      Chat.loadAll().then(chats => UI.renderChatList(chats));
    } catch (e) { showToast(e.message); }
  }
  async function doRegister() {
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPassword").value;
    const name = document.getElementById("loginName").value.trim() || email.split("@")[0];
    if (!email || !password) return showToast("Please fill email and password");
    try {
      await register(email, password, name);
      showToast("Registered! Check your email for verification code.");
    } catch (e) { showToast(e.message); }
  }
  async function forgotPassword() {
    const email = document.getElementById("forgotEmail").value.trim();
    if (!email) return showToast("Enter email");
    const res = await fetch(`${CONFIG.API_BASE}/api/forgot-password?email=${encodeURIComponent(email)}`, { method: "POST" });
    const data = await res.json();
    showToast(data.message);
    UI.closeForgotModal();
  }
  async function verifyEmail() {
    const email = document.getElementById("verifyEmail").value.trim();
    const code = document.getElementById("verifyCode").value.trim();
    if (!email || !code) return showToast("Please fill email and code");
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/verify-email?email=${encodeURIComponent(email)}&code=${code}`, { method: "POST" });
      const data = await res.json();
      showToast(data.message);
      UI.closeVerifyModal();
    } catch (e) { showToast(e.message); }
  }

  return { restore, login, register, logout, getToken, getUser, showLoginModal, closeLoginModal, doLogin, doRegister, forgotPassword, verifyEmail };
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
  let currentChatId = null, abortController = null;

  async function newChat() {
    if (!Auth.getToken()) return showToast("Please login first");
    const title = prompt("Chat title", "New Chat") || "New Chat";
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/chats`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      if (!res.ok) throw new Error("Failed to create chat");
      const chat = await res.json();
      currentChatId = chat.id;
      await loadAll().then(chats => UI.renderChatList(chats));
      await load(chat.id);
    } catch (e) { showToast(e.message); }
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
    const data = await res.json();
    currentChatId = data.id;
    UI.renderMessages(data.messages);
    return data;
  }

  async function deleteChat(chatId) {
    await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${Auth.getToken()}` }
    });
    if (currentChatId === chatId) {
      currentChatId = null;
      document.getElementById("reply").innerHTML = '<div class="welcome"><h1>AgentOS AI</h1></div>';
    }
    await loadAll().then(chats => UI.renderChatList(chats));
  }

  async function updateChat(chatId, updates) {
    await fetchWithTimeout(`${CONFIG.API_BASE}/api/chats/${chatId}`, {
      method: "PUT",
      headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
      body: JSON.stringify(updates)
    });
    await loadAll().then(chats => UI.renderChatList(chats));
  }

  async function sendMessage(text, model, tool, mode) {
    if (!Auth.getToken()) return showToast("Please login first");
    if (!currentChatId) { await newChat(); if (!currentChatId) return; }
    const replyEl = document.getElementById("reply");
    if (!replyEl) return;

    UI.showLoading();
    const userDiv = document.createElement("div");
    userDiv.className = "message user-msg";
    userDiv.textContent = text;
    replyEl.appendChild(userDiv);
    replyEl.scrollTop = replyEl.scrollHeight;
    document.getElementById("msg").value = "";

    const chatData = await load(currentChatId);
    const messages = chatData.messages.map(m => ({ role: m.role, content: m.content }));
    messages.push({ role: "user", content: text });

    abortController = new AbortController();
    try {
      const streamRes = await fetchWithTimeout(`${CONFIG.API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: currentChatId, messages, model, tool, mode }),
        signal: abortController.signal
      }, 180000);
      if (!streamRes.ok) throw new Error("Stream failed");
      UI.hideLoading();
      const reader = streamRes.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      const assistantDiv = document.createElement("div");
      assistantDiv.className = "message assistant-msg";
      replyEl.appendChild(assistantDiv);

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) {
                assistantContent += parsed.content;
                const finalHtml = UI.formatMessageContent(assistantContent);
                assistantDiv.innerHTML = DOMPurify.sanitize(finalHtml, {
                  ALLOWED_TAGS: ["b","i","strong","em","pre","code","span","br"],
                  ALLOWED_ATTR: ["class"]
                });
                replyEl.scrollTop = replyEl.scrollHeight;
              }
            } catch(e) {}
          }
        }
      }
      const updatedChat = await load(currentChatId);
      if (updatedChat) UI.renderMessages(updatedChat.messages);
    } catch (e) {
      UI.hideLoading();
      if (e.name !== "AbortError") showToast("Error: " + e.message);
    }
  }

  function stopGeneration() {
    if (abortController) { abortController.abort(); abortController = null; }
  }

  async function exportChats() {
    const chats = await loadAll();
    const fullExport = [];
    for (const chat of chats) {
      const res = await fetch(`${CONFIG.API_BASE}/api/chats/${chat.id}/export`, {
        headers: { "Authorization": `Bearer ${Auth.getToken()}` }
      });
      if (res.ok) fullExport.push(await res.json());
    }
    const blob = new Blob([JSON.stringify(fullExport, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "agentos_chats.json";
    a.click();
  }

  function importChats() { document.getElementById("importInput").click(); }

  async function handleImport(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const data = JSON.parse(e.target.result);
        for (const item of data) {
          const { chat, messages } = item;
          const res = await fetch(`${CONFIG.API_BASE}/api/chats`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
            body: JSON.stringify({ title: chat.title })
          });
          if (!res.ok) continue;
          const newChat = await res.json();
          for (const msg of messages) {
            await fetch(`${CONFIG.API_BASE}/api/chats/${newChat.id}/messages`, {
              method: "POST",
              headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
              body: JSON.stringify({ role: msg.role, content: msg.content })
            });
          }
          if (chat.pinned) await updateChat(newChat.id, { pinned: true });
        }
        await loadAll().then(chats => UI.renderChatList(chats));
        showToast("Import complete!");
      } catch (err) { showToast("Import failed: " + err.message); }
    };
    reader.readAsText(file);
  }

  return { newChat, loadAll, load, deleteChat, updateChat, sendMessage, stopGeneration, exportChats, importChats, handleImport, get currentChatId() { return currentChatId; }, set currentChatId(id) { currentChatId = id; } };
})();

// ===================== UI =====================
const UI = (() => {
  let selectedChatId = null;

  function formatMessageContent(text) {
    // Simple markdown: code blocks and line breaks
    return text.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
      return `<pre class="code-block"><code>${code.replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code></pre>`;
    }).replace(/\n/g, "<br>");
  }

  function renderMessages(messages) {
    const chatArea = document.getElementById("reply");
    if (!chatArea) return;
    chatArea.innerHTML = "";
    messages.forEach(msg => {
      const div = document.createElement("div");
      div.className = `message ${msg.role === "user" ? "user-msg" : "assistant-msg"}`;
      const raw = msg.content || "";
      const formatted = formatMessageContent(raw);
      div.innerHTML = DOMPurify.sanitize(formatted, {
        ALLOWED_TAGS: ["b","i","strong","em","pre","code","span","br"],
        ALLOWED_ATTR: ["class"]
      });
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
      div.dataset.id = chat.id;
      div.innerHTML = `<span class="chat-item-title">${chat.pinned ? '📌 ' : ''}💬 ${chat.title}</span><button class="chat-item-delete">🗑</button>`;
      div.querySelector(".chat-item-title").addEventListener("click", () => Chat.load(chat.id));
      div.querySelector(".chat-item-delete").addEventListener("click", (e) => { e.stopPropagation(); Chat.deleteChat(chat.id); });
      div.addEventListener("contextmenu", (e) => { e.preventDefault(); showContextMenu(e, chat.id); });
      list.appendChild(div);
    });
  }

  function showContextMenu(e, chatId) {
    selectedChatId = chatId;
    const menu = document.getElementById("chatMenu");
    menu.classList.add("show");
    const rect = menu.getBoundingClientRect();
    let top = e.pageY, left = e.pageX;
    if (top + rect.height > window.innerHeight) top = window.innerHeight - rect.height - 10;
    if (left + rect.width > window.innerWidth) left = window.innerWidth - rect.width - 10;
    menu.style.left = left + "px";
    menu.style.top = top + "px";
  }

  function pinChat() {
    if (!selectedChatId) return;
    Chat.load(selectedChatId).then(chat => {
      if (chat) {
        const newPinned = !chat.pinned;
        Chat.updateChat(selectedChatId, { pinned: newPinned }).then(() => {
          Chat.loadAll().then(chats => renderChatList(chats));
        });
      }
    });
    document.getElementById("chatMenu").classList.remove("show");
  }

  function renameChat() {
    if (!selectedChatId) return;
    document.getElementById("renameInput").value = "";
    document.getElementById("renameModal").classList.remove("hidden");
    document.getElementById("chatMenu").classList.remove("show");
  }

  function submitRename() {
    const newTitle = document.getElementById("renameInput").value.trim();
    if (newTitle && selectedChatId) {
      Chat.updateChat(selectedChatId, { title: newTitle });
    }
    document.getElementById("renameModal").classList.add("hidden");
  }

  function closeRenameModal() { document.getElementById("renameModal").classList.add("hidden"); }

  function deleteChat() {
    if (selectedChatId) Chat.deleteChat(selectedChatId);
    document.getElementById("chatMenu").classList.remove("show");
  }

  function updateAuthUI() {
    const user = Auth.getUser();
    document.getElementById("accountStatus").textContent = user ? user.name : "Guest";
    document.getElementById("founderPanel").style.display = (user && CONFIG.FOUNDER_EMAILS.includes(user.email.toLowerCase())) ? "block" : "none";
  }

  function searchChats() {
    const query = document.getElementById("searchChat").value.toLowerCase();
    document.querySelectorAll(".chat-item").forEach(item => {
      item.style.display = item.textContent.toLowerCase().includes(query) ? "" : "none";
    });
  }

  function searchMessages() {
    const query = document.getElementById("msgSearch")?.value.toLowerCase();
    if (!query) {
      document.querySelectorAll("#reply .message").forEach(el => el.style.display = "");
      return;
    }
    document.querySelectorAll("#reply .message").forEach(el => {
      el.style.display = el.textContent.toLowerCase().includes(query) ? "" : "none";
    });
  }

  function toggleSidebar() { document.getElementById("sidebar").classList.toggle("active"); }
  function showLoginModal() { document.getElementById("loginModal").classList.remove("hidden"); }
  function closeLoginModal() { document.getElementById("loginModal").classList.add("hidden"); }
  function showSettingsModal() { document.getElementById("settingsModal").classList.remove("hidden"); }
  function closeSettingsModal() { document.getElementById("settingsModal").classList.add("hidden"); }
  function showMemoryPanel() { document.getElementById("memoryModal").classList.remove("hidden"); Memory.loadMemory(); }
  function closeMemoryPanel() { document.getElementById("memoryModal").classList.add("hidden"); }
  function showProfileModal() {
    const user = Auth.getUser();
    if (!user) return showToast("Please login first");
    document.getElementById("profileName").value = user.name || "";
    document.getElementById("profileAvatarUrl").value = user.avatar_url || "";
    const avatar = document.getElementById("profileAvatar");
    const fallback = document.getElementById("profileAvatarFallback");
    if (user.avatar_url) {
      avatar.src = user.avatar_url;
      avatar.style.display = "block";
      fallback.style.display = "none";
    } else {
      avatar.style.display = "none";
      fallback.textContent = (user.name || "?").charAt(0).toUpperCase();
      fallback.style.display = "flex";
    }
    document.getElementById("profileModal").classList.remove("hidden");
  }
  function closeProfileModal() { document.getElementById("profileModal").classList.add("hidden"); }
  function showForgotPassword() {
    document.getElementById("forgotModal").classList.remove("hidden");
    document.getElementById("loginModal").classList.add("hidden");
  }
  function closeForgotModal() { document.getElementById("forgotModal").classList.add("hidden"); }
  function showVerifyEmail() {
    document.getElementById("verifyModal").classList.remove("hidden");
    document.getElementById("loginModal").classList.add("hidden");
  }
  function closeVerifyModal() { document.getElementById("verifyModal").classList.add("hidden"); }

  function openAttachmentMenu() { document.getElementById("attachmentMenu").classList.toggle("active"); }
  function closeAttachmentMenu() { document.getElementById("attachmentMenu").classList.remove("active"); }

  function showLoading() {
    const replyEl = document.getElementById("reply");
    if (!replyEl) return;
    const skeleton = document.createElement("div");
    skeleton.className = "skeleton";
    skeleton.id = "loadingSkeleton";
    replyEl.appendChild(skeleton);
  }
  function hideLoading() {
    const el = document.getElementById("loadingSkeleton");
    if (el) el.remove();
  }

  return { renderMessages, renderChatList, showContextMenu, pinChat, renameChat, submitRename, closeRenameModal, deleteChat, updateAuthUI, searchChats, searchMessages, toggleSidebar, showLoginModal, closeLoginModal, showSettingsModal, closeSettingsModal, showMemoryPanel, closeMemoryPanel, showProfileModal, closeProfileModal, showForgotPassword, closeForgotModal, showVerifyEmail, closeVerifyModal, openAttachmentMenu, closeAttachmentMenu, showLoading, hideLoading, formatMessageContent };
})();

// ===================== Voice =====================
const Voice = (() => {
  let speechActive = true, recognition = null;

  function toggleVoice() {
    speechActive = !speechActive;
    document.getElementById("voiceStatus").textContent = speechActive ? "ON" : "OFF";
    document.getElementById("speechBtn").textContent = speechActive ? "🔊" : "🔇";
  }

  function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return showToast("Voice not supported");
    recognition = new SR();
    recognition.lang = "en-US";
    recognition.start();
    document.getElementById("msg").placeholder = "Listening...";
    recognition.onresult = (e) => {
      document.getElementById("msg").value = e.results[0][0].transcript;
      document.getElementById("msg").placeholder = "Message AgentOS AI...";
    };
    recognition.onerror = () => { document.getElementById("msg").placeholder = "Message AgentOS AI..."; };
  }

  function speak(text) {
    if (!speechActive || !text) return;
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text.replace(/<[^>]*>/g, ""));
    synth.speak(utterance);
  }

  return { toggleVoice, startListening, speak };
})();

// ===================== Memory =====================
const Memory = (() => {
  async function loadMemory() {
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/memory`, { headers: { Authorization: `Bearer ${Auth.getToken()}` } });
      if (res.ok) {
        const data = await res.json();
        const list = document.getElementById("memoryList");
        list.innerHTML = "";
        data.items.forEach(item => {
          const div = document.createElement("div");
          div.innerHTML = `<strong>${item.key}</strong>: ${item.value} <button onclick="Memory.deleteMemory('${item.key}')">✕</button>`;
          list.appendChild(div);
        });
      }
    } catch (e) { showToast("Error loading memory"); }
  }

  async function addMemory() {
    const key = document.getElementById("memoryKey").value.trim();
    const value = document.getElementById("memoryValue").value.trim();
    if (!key || !value) return;
    await fetch(`${CONFIG.API_BASE}/api/memory?key=${encodeURIComponent(key)}&value=${encodeURIComponent(value)}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${Auth.getToken()}` }
    });
    loadMemory();
    document.getElementById("memoryKey").value = "";
    document.getElementById("memoryValue").value = "";
  }

  async function deleteMemory(key) {
    await fetch(`${CONFIG.API_BASE}/api/memory/${encodeURIComponent(key)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${Auth.getToken()}` }
    });
    loadMemory();
  }

  function toggleMemory() {
    const current = document.getElementById("memoryStatus").textContent === "ON";
    const newState = !current;
    document.getElementById("memoryStatus").textContent = newState ? "ON" : "OFF";
    Settings.update("memory_enabled", newState ? 1 : 0);
  }

  return { loadMemory, addMemory, deleteMemory, toggleMemory };
})();

// ===================== Settings =====================
const Settings = (() => {
  async function update(key, value) {
    const body = {};
    body[key] = value;
    await fetch(`${CONFIG.API_BASE}/api/settings`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (key === "theme") {
      document.body.classList.toggle("light", value === "light");
      document.getElementById("themeStatus").textContent = value === "light" ? "Light" : "Dark";
    }
    if (key === "speech_active") {
      const state = value ? true : false;
      document.getElementById("voiceStatus").textContent = state ? "ON" : "OFF";
      document.getElementById("speechBtn").textContent = state ? "🔊" : "🔇";
    }
    if (key === "memory_enabled") {
      document.getElementById("memoryStatus").textContent = value ? "ON" : "OFF";
    }
    // other keys handled by UI directly
  }

  function toggleAutoSave() {
    const current = document.getElementById("saveStatus").textContent === "ON";
    const newState = !current;
    document.getElementById("saveStatus").textContent = newState ? "ON" : "OFF";
    update("auto_save", newState ? 1 : 0);
  }

  function toggleAutoName() {
    const current = document.getElementById("autoNameStatus").textContent === "ON";
    const newState = !current;
    document.getElementById("autoNameStatus").textContent = newState ? "ON" : "OFF";
    update("auto_name", newState ? 1 : 0);
  }

  return { update, toggleAutoSave, toggleAutoName };
})();

// ===================== Profile =====================
const Profile = (() => {
  async function save() {
    const name = document.getElementById("profileName").value.trim();
    const avatar = document.getElementById("profileAvatarUrl").value.trim();
    const body = {};
    if (name) body.name = name;
    if (avatar) body.avatar_url = avatar;
    if (Object.keys(body).length === 0) return;
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/profile`, {
        method: "PUT",
        headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (res.ok) {
        showToast("Profile updated");
        const user = Auth.getUser();
        if (name) user.name = name;
        if (avatar) user.avatar_url = avatar;
        UI.updateAuthUI();
        UI.closeProfileModal();
      }
    } catch (e) { showToast(e.message); }
  }

  function load() {
    const user = Auth.getUser();
    if (!user) return;
    const fallback = document.getElementById("profileAvatarFallback");
    if (fallback) fallback.textContent = (user.name || "?").charAt(0).toUpperCase();
  }

  return { save, load };
})();

// ===================== App Core =====================
const App = (() => {
  let selectedModel = "GPT-4o", selectedTool = "No Tool", currentMode = "chat";

  async function init() {
    await Database.open();
    await Auth.restore();
    UI.updateAuthUI();

    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/settings`, { headers: { Authorization: `Bearer ${Auth.getToken()}` } });
      if (res.ok) {
        const s = await res.json();
        document.body.classList.toggle("light", s.theme === "light");
        document.getElementById("themeStatus").textContent = s.theme === "light" ? "Light" : "Dark";
        document.getElementById("voiceStatus").textContent = s.speech_active ? "ON" : "OFF";
        document.getElementById("memoryStatus").textContent = s.memory_enabled ? "ON" : "OFF";
        document.getElementById("saveStatus").textContent = s.auto_save ? "ON" : "OFF";
        document.getElementById("autoNameStatus").textContent = s.auto_name ? "ON" : "OFF";
      }
    } catch(e) {}

    selectedModel = localStorage.getItem("selectedModel") || "GPT-4o";
    document.getElementById("activeModel").textContent = selectedModel;
    selectedTool = localStorage.getItem("selectedTool") || "No Tool";
    document.getElementById("activeTool").textContent = selectedTool;
    highlightModelButton(selectedModel);
    highlightToolButton(selectedTool);

    if (Auth.getToken()) {
      try { const chats = await Chat.loadAll(); UI.renderChatList(chats); } catch(e) {}
    }

    document.getElementById("msg").addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    const langSelect = document.getElementById("langSelect");
    if (langSelect) CONFIG.ALL_LANGUAGES.forEach(l => { const o = document.createElement("option"); o.value = l; o.textContent = l; langSelect.appendChild(o); });

    ["cameraInput","galleryInput","fileInput"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("change", e => showFilePreviews(e.target.files));
    });

    setInterval(() => { document.getElementById("currentDateTime").textContent = new Date().toLocaleString(); }, 1000);
  }

  function switchMode(mode, btn) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    document.querySelectorAll(".code-mode-ui").forEach(el => el.style.display = "none");
    const input = document.getElementById("msg");
    const ph = { chat:"Message AgentOS AI...", code:"Describe code...", teacher:"What do you want to learn?", image:"Describe the image...", voice:"Click microphone...", search:"Search the web...", cybersecurity:"Explore cybersecurity courses..." };
    if (input) input.placeholder = ph[mode] || "Message AgentOS AI...";
    if (mode === "code" || mode === "teacher") document.getElementById("codeModeUI").style.display = "flex";
    if (mode === "image") document.getElementById("imageGenPanel").style.display = "flex";
    if (mode === "voice") document.getElementById("voicePanel").style.display = "flex";
    if (mode === "search") document.getElementById("searchPanel").style.display = "flex";
    if (mode === "cybersecurity") { document.getElementById("cybersecPanel").style.display = "block"; loadCybersecCourses(); }
    // Show message search bar in chat mode
    document.getElementById("msgSearchBar").style.display = (mode === "chat") ? "flex" : "none";
  }

  function selectModel(model, btn) {
    selectedModel = model;
    localStorage.setItem("selectedModel", model);
    document.querySelectorAll(".model-chip").forEach(b => b.classList.remove("active-model"));
    if (btn) btn.classList.add("active-model");
    document.getElementById("activeModel").textContent = model;
  }

  function selectTool(tool, btn) {
    selectedTool = (selectedTool === tool) ? "No Tool" : tool;
    localStorage.setItem("selectedTool", selectedTool);
    document.querySelectorAll(".tool-btn").forEach(b => b.classList.remove("active-tool"));
    if (selectedTool !== "No Tool" && btn) btn.classList.add("active-tool");
    document.getElementById("activeTool").textContent = selectedTool;
  }

  function changeLanguage(lang) { localStorage.setItem("selectedLanguage", lang); document.getElementById("langSelectGlobal").value = lang; }

  async function sendMessage() {
    const text = document.getElementById("msg").value.trim();
    if (!text) return;
    Chat.sendMessage(text, selectedModel, selectedTool, currentMode);
  }

  function stopGeneration() { Chat.stopGeneration(); }

  async function generateImage() {
    const prompt = document.getElementById("imagePrompt").value.trim();
    if (!prompt) return;
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/image-generate`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${Auth.getToken()}`, "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      if (res.ok) {
        const data = await res.json();
        const img = document.createElement("img");
        img.src = data.url;
        img.style.maxWidth = "100%";
        document.getElementById("reply").appendChild(img);
      }
    } catch (e) { showToast("Image generation failed"); }
  }

  async function webSearch() {
    const query = document.getElementById("searchQuery").value.trim();
    if (!query) return;
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/search?q=${encodeURIComponent(query)}`, { headers: { "Authorization": `Bearer ${Auth.getToken()}` } });
      if (res.ok) {
        const data = await res.json();
        const replyEl = document.getElementById("reply");
        if (replyEl) {
          data.results.forEach(r => {
            const div = document.createElement("div");
            div.className = "message assistant-msg";
            div.innerHTML = `<strong>${r.title}</strong><br>${r.snippet}<br><a href="${r.url}" target="_blank">Read more</a>`;
            replyEl.appendChild(div);
          });
        }
      }
    } catch (e) { showToast("Search failed"); }
  }

  async function loadCybersecCourses() {
    try {
      const res = await fetch(`${CONFIG.API_BASE}/api/cybersecurity/courses`);
      if (res.ok) {
        const courses = await res.json();
        document.getElementById("cybersecContent").innerHTML = courses.map(c =>
          `<div style="margin-bottom:10px;padding:8px;background:#1e293b;border-radius:8px;">
            <strong>${c.title}</strong> (${c.level})<br><small>${c.provider}</small><br>
            <a href="${c.url}" target="_blank" style="color:#60a5fa;">Go to course</a></div>`
        ).join("");
      }
    } catch (e) { showToast("Failed to load courses"); }
  }

  function toggleTheme() {
    document.body.classList.toggle("light");
    const isLight = document.body.classList.contains("light");
    document.getElementById("themeStatus").textContent = isLight ? "Light" : "Dark";
    Settings.update("theme", isLight ? "light" : "dark");
  }

  function highlightModelButton(model) {
    document.querySelectorAll(".model-chip").forEach(btn => {
      btn.classList.remove("active-model");
      if (btn.dataset.model === model) btn.classList.add("active-model");
    });
  }
  function highlightToolButton(tool) {
    document.querySelectorAll(".tool-btn").forEach(btn => {
      btn.classList.remove("active-tool");
      if (btn.dataset.tool === tool) btn.classList.add("active-tool");
    });
  }

  return { init, switchMode, selectModel, selectTool, changeLanguage, sendMessage, stopGeneration, generateImage, webSearch, loadCybersecCourses, toggleTheme };
})();

// ===================== Admin =====================
const Admin = (() => {
  async function listUsers() {
    const res = await fetch(`${CONFIG.API_BASE}/api/admin/users`, { headers: { "Authorization": `Bearer ${Auth.getToken()}` } });
    if (!res.ok) return showToast("Access denied");
    const users = await res.json();
    alert(users.map(u => `${u.email} - ${u.role} [${u.status}]`).join("\n"));
  }
  async function banUser() { const e = prompt("Email:"); if (!e) return; await fetch(`${CONFIG.API_BASE}/api/admin/users/${e}/ban`, { method: "PUT", headers: { "Authorization": `Bearer ${Auth.getToken()}` } }); showToast("Done"); }
  async function suspendUser() { const e = prompt("Email:"); if (!e) return; await fetch(`${CONFIG.API_BASE}/api/admin/users/${e}/suspend`, { method: "PUT", headers: { "Authorization": `Bearer ${Auth.getToken()}` } }); showToast("Done"); }
  async function unbanUser() { const e = prompt("Email:"); if (!e) return; await fetch(`${CONFIG.API_BASE}/api/admin/users/${e}/unban`, { method: "PUT", headers: { "Authorization": `Bearer ${Auth.getToken()}` } }); showToast("Done"); }
  async function deleteUser() { const e = prompt("Email to delete:"); if (!e || !confirm("Permanently delete?")) return; await fetch(`${CONFIG.API_BASE}/api/admin/users/${e}`, { method: "DELETE", headers: { "Authorization": `Bearer ${Auth.getToken()}` } }); showToast("Deleted"); }
  async function listChats() {
    const res = await fetch(`${CONFIG.API_BASE}/api/admin/chats`, { headers: { "Authorization": `Bearer ${Auth.getToken()}` } });
    if (!res.ok) return showToast("Access denied");
    const chats = await res.json();
    alert(chats.map(c => `[${c.id}] ${c.user_email}: ${c.title}`).join("\n"));
  }
  async function logs() {
    const res = await fetch(`${CONFIG.API_BASE}/api/admin/logs`, { headers: { "Authorization": `Bearer ${Auth.getToken()}` } });
    if (!res.ok) return showToast("Access denied");
    const logs = await res.json();
    alert(logs.map(l => `[${l.timestamp}] ${l.admin_email} ${l.action} ${l.target}`).join("\n"));
  }
  async function stats() {
    const res = await fetch(`${CONFIG.API_BASE}/api/admin/stats`, { headers: { "Authorization": `Bearer ${Auth.getToken()}` } });
    if (!res.ok) return showToast("Access denied");
    const data = await res.json();
    alert(`Users: ${data.total_users} (Active: ${data.active_users})\nChats: ${data.total_chats}\nMessages: ${data.total_messages}`);
  }
  return { listUsers, banUser, suspendUser, unbanUser, deleteUser, listChats, logs, stats };
})();

// ===================== Attachments =====================
let uploadedFiles = [];
function showFilePreviews(files) {
  if (files && files.length) uploadedFiles.push(...Array.from(files));
  renderAttachments();
}
function renderAttachments() {
  const preview = document.getElementById("attachmentPreview");
  if (!preview) return;
  preview.innerHTML = "";
  uploadedFiles.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.textContent = file.name;
    const btn = document.createElement("span");
    btn.textContent = " ✕"; btn.style.cursor = "pointer";
    btn.onclick = () => { uploadedFiles.splice(index, 1); renderAttachments(); };
    chip.appendChild(btn);
    preview.appendChild(chip);
  });
}
const Attachments = {
  openCamera: () => { document.getElementById("cameraInput").click(); UI.closeAttachmentMenu(); },
  openGallery: () => { document.getElementById("galleryInput").click(); UI.closeAttachmentMenu(); },
  openFiles: () => { document.getElementById("fileInput").click(); UI.closeAttachmentMenu(); }
};

// ===================== Helpers =====================
function showToast(msg) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ===================== Start =====================
window.addEventListener("load", () => App.init().catch(e => console.error(e)));