package com.example.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.api.GeminiClient
import com.example.data.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class TaskViewModel(private val repository: AgentRepository) : ViewModel() {

    // --- Authentication & Current Session User State ---
    private val _currentUser = MutableStateFlow<User?>(null)
    val currentUser: StateFlow<User?> = _currentUser.asStateFlow()

    private val _authError = MutableStateFlow<String?>(null)
    val authError: StateFlow<String?> = _authError.asStateFlow()

    private val _authSuccess = MutableStateFlow<String?>(null)
    val authSuccess: StateFlow<String?> = _authSuccess.asStateFlow()


    // --- Active Chat Selection Flows ---
    private val _selectedChat = MutableStateFlow<ChatSession?>(null)
    val selectedChat: StateFlow<ChatSession?> = _selectedChat.asStateFlow()

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    // Chats List depending on Logged-In User ID
    val allChats: StateFlow<List<ChatSession>> = _currentUser
        .flatMapLatest { user ->
            if (user == null) {
                flowOf(emptyList())
            } else {
                _searchQuery.flatMapLatest { query ->
                    if (query.isBlank()) {
                        repository.getChatsForUser(user.id)
                    } else {
                        repository.searchChatsForUser(user.id, query)
                    }
                }
            }
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )


    // Messages list of currently selected ChatSession
    val chatMessages: StateFlow<List<ChatMessage>> = _selectedChat
        .flatMapLatest { chat ->
            if (chat == null) {
                flowOf(emptyList())
            } else {
                repository.getMessagesForChat(chat.id)
            }
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )


    // Long-Term memory entries
    val memories: StateFlow<List<MemoryItem>> = _currentUser
        .flatMapLatest { user ->
            if (user == null) flowOf(emptyList()) else repository.getMemoriesForUser(user.id)
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )


    // --- System Audit Logs (Founder View) ---
    val systemEvents: StateFlow<List<AppEvent>> = repository.allEvents
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )


    // --- Platform Analytics (Founder View) ---
    val analytics: StateFlow<AnalyticsData?> = repository.analytics
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = null
        )


    // --- User Management List (Admin/Founder View) ---
    val allUsersList: StateFlow<List<User>> = repository.allUsers
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    // --- System Configuration Settings (Founder View) ---
    private val _isMaintenanceMode = MutableStateFlow(false)
    val isMaintenanceMode: StateFlow<Boolean> = _isMaintenanceMode.asStateFlow()

    private val _safetyFilterLevel = MutableStateFlow("Standard Safety")
    val safetyFilterLevel: StateFlow<String> = _safetyFilterLevel.asStateFlow()

    private val _allowedRegistration = MutableStateFlow(true)
    val allowedRegistration: StateFlow<Boolean> = _allowedRegistration.asStateFlow()

    private val _apiUsageLimit = MutableStateFlow(250)
    val apiUsageLimit: StateFlow<Int> = _apiUsageLimit.asStateFlow()

    private val _systemPromptPrefix = MutableStateFlow("AgentOS Secure Mode Activated.")
    val systemPromptPrefix: StateFlow<String> = _systemPromptPrefix.asStateFlow()


    // --- Active Message Sending & Generation States ---
    private val _selectedModel = MutableStateFlow("GPT-4o")
    val selectedModel: StateFlow<String> = _selectedModel.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private var activeGenerationJob: Job? = null


    // --- Attachment simulation states ---
    private val _attachedImageBase64 = MutableStateFlow<String?>(null)
    val attachedImageBase64: StateFlow<String?> = _attachedImageBase64.asStateFlow()

    private val _attachedImageLabel = MutableStateFlow<String?>(null)
    val attachedImageLabel: StateFlow<String?> = _attachedImageLabel.asStateFlow()


    // Initializer to seed and check session status
    init {
        viewModelScope.launch {
            repository.seedDatabase()
        }
    }


    // --- Authentication Business Logic ---

    fun login(email: String, pass: String) {
        viewModelScope.launch {
            _authError.value = null
            _authSuccess.value = null
            val sanitizedEmail = email.trim().lowercase()
            
            if (sanitizedEmail.isBlank() || pass.isBlank()) {
                _authError.value = "Credentials cannot be blank"
                repository.logEvent("SECURITY_AUDIT", "Failed login attempt: blank credentials entered")
                return@launch
            }

            val user = repository.getUserByEmail(sanitizedEmail)
            if (user == null) {
                _authError.value = "User account not found"
                repository.logEvent("SECURITY_AUDIT", "Rejected login: account $sanitizedEmail not registered")
                return@launch
            }

            if (!PasswordHasher.verify(pass, user.passwordHash)) { // Secure verification using BCrypt hashing
                _authError.value = "Invalid password. Access Denied."
                repository.logEvent("SECURITY_AUDIT", "Rejected login: incorrect password entered for $sanitizedEmail")
                return@launch
            }

            if (user.status == "banned" || user.status == "suspended") {
                _authError.value = "Access suspended. This account is banned from AgentOS."
                repository.logEvent("SECURITY_AUDIT", "Blocked banned user login: $sanitizedEmail")
                return@launch
            }

            if (_isMaintenanceMode.value && user.role != "founder") {
                _authError.value = "System is in maintenance. Access restricted to Founder Rahul Kumar Mahto."
                repository.logEvent("SECURITY_AUDIT", "Blocked user login during maintenance: $sanitizedEmail")
                return@launch
            }

            // Session success
            _currentUser.value = user
            _authSuccess.value = "Welcome back, ${user.username}!"
            repository.logEvent("AUTH", "User successfully authenticated: ${user.username}")
        }
    }

    fun register(username: String, email: String, pass: String) {
        viewModelScope.launch {
            _authError.value = null
            _authSuccess.value = null
            
            if (!_allowedRegistration.value) {
                _authError.value = "New container registrations are temporarily suspended by Founder Rahul Kumar Mahto."
                repository.logEvent("SECURITY_AUDIT", "Blocked public check-in. Registrations closed.")
                return@launch
            }

            if (username.isBlank() || email.isBlank() || pass.isBlank()) {
                _authError.value = "All registration parameters are mandatory."
                return@launch
            }

            val sanitizedEmail = email.trim().lowercase()
            val existing = repository.getUserByEmail(sanitizedEmail)
            if (existing != null) {
                _authError.value = "Email is already registered on this platform."
                return@launch
            }

            val u = User(
                username = username.trim(),
                email = sanitizedEmail,
                passwordHash = PasswordHasher.hash(pass), // Secure encryption of user login keys
                role = "user",
                status = "active"
            )
            repository.insertUser(u)
            _authSuccess.value = "Account built successfully! Please log in."
            repository.logEvent("AUTH", "Registered new client: ${username.trim()}")
        }
    }

    fun logout() {
        viewModelScope.launch {
            val name = _currentUser.value?.username ?: "Guest"
            repository.logEvent("AUTH", "User signed out: $name")
            _currentUser.value = null
            _selectedChat.value = null
            _authSuccess.value = "Signed out successfully."
        }
    }

    fun guestLogin() {
        viewModelScope.launch {
            _authError.value = null
            _authSuccess.value = null
            
            // Build a non-conflicting unique guest account registered cleanly into the DB
            val uniqueSuffix = System.currentTimeMillis()
            val guestUser = User(
                username = "Guest Operator",
                email = "guest_$uniqueSuffix@agentos.ai",
                passwordHash = PasswordHasher.hash("guest_bypass_$uniqueSuffix"),
                role = "user",
                status = "active"
            )
            val insertedId = repository.insertUser(guestUser)
            val loggedInGuest = guestUser.copy(id = insertedId.toInt())
            
            _currentUser.value = loggedInGuest
            _authSuccess.value = "Logged in under secure Guest account."
            repository.logEvent("AUTH", "Guest access initiated with unique DB identifier $insertedId.")
        }
    }

    fun triggerPasswordReset(email: String) {
        viewModelScope.launch {
            _authError.value = null
            _authSuccess.value = null
            val sanitized = email.trim().lowercase()
            val u = repository.getUserByEmail(sanitized)
            if (u == null) {
                _authError.value = "No registered profile linked with $email"
                return@launch
            }
            _authSuccess.value = "Password reset dispatch link forwarded to $email."
            repository.logEvent("AUTH", "Password reset request filed for $email")
        }
    }


    // --- Sidebar Chat Controls ---

    fun createChatSession(customTitle: String? = null) {
        val user = _currentUser.value ?: return
        viewModelScope.launch {
            val newChat = ChatSession(
                userId = user.id,
                title = customTitle ?: "New AI ChatSession",
                modelId = _selectedModel.value
            )
            val id = repository.insertChat(newChat)
            val fetched = repository.getChatById(id.toInt())
            if (fetched != null) {
                _selectedChat.value = fetched
                
                // Set default system welcome message
                repository.insertMessage(
                    ChatMessage(
                        chatId = fetched.id,
                        role = "assistant",
                        content = "Welcome to AgentOS! Operating model: **${_selectedModel.value}**. Let me know what to construct or analyze, Rahul!"
                    )
                )
            }
        }
    }

    fun selectChat(chat: ChatSession?) {
        _selectedChat.value = chat
    }

    fun renameChat(chat: ChatSession, newTitle: String) {
        if (newTitle.isBlank()) return
        viewModelScope.launch {
            val updated = chat.copy(title = newTitle.trim())
            repository.updateChat(updated)
            if (_selectedChat.value?.id == chat.id) {
                _selectedChat.value = updated
            }
        }
    }

    fun pinChat(chat: ChatSession) {
        viewModelScope.launch {
            val updated = chat.copy(isPinned = !chat.isPinned)
            repository.updateChat(updated)
            if (_selectedChat.value?.id == chat.id) {
                _selectedChat.value = updated
            }
        }
    }

    fun deleteChat(chat: ChatSession) {
        viewModelScope.launch {
            repository.deleteChatById(chat.id)
            if (_selectedChat.value?.id == chat.id) {
                _selectedChat.value = null
            }
        }
    }

    fun setSearchQuery(query: String) {
        _searchQuery.value = query
    }

    fun changeModelSelection(modelName: String) {
        _selectedModel.value = modelName
        // If current chat is selected, update its model label
        val current = _selectedChat.value
        if (current != null) {
            viewModelScope.launch {
                val updated = current.copy(modelId = modelName)
                repository.updateChat(updated)
                _selectedChat.value = updated
            }
        }
    }


    // --- Live AI Chat Generation Pipeline ---

    fun sendMessage(text: String) {
        val user = _currentUser.value ?: return
        val currentChat = _selectedChat.value ?: return
        if (text.isBlank() && _attachedImageBase64.value == null) return

        viewModelScope.launch {
            // Save User's prompt
            val uMsg = ChatMessage(
                chatId = currentChat.id,
                role = "user",
                content = text.trim(),
                imageUri = _attachedImageLabel.value
            )
            repository.insertMessage(uMsg)

            _isGenerating.value = true

            // Gather attachments and clear attachment buffer
            val base64 = _attachedImageBase64.value
            val currentLabel = _attachedImageLabel.value
            clearAttachment()

            // Construct customized prompt & system instructions
            val activeModel = currentChat.modelId
            val systemPrompt = buildSystemPrompt(activeModel)

            // Dynamic Image vs. Text selection
            val prompt = if (currentLabel != null) {
                "[Attached Object: $currentLabel]. Analyze the attached entity: ${text.trim()}"
            } else {
                text.trim()
            }

            // Launch Cancellable LLM Generation coroutine
            activeGenerationJob = launch(Dispatchers.IO) {
                val reply = try {
                    GeminiClient.generateChatResponse(prompt, activeModel, systemPrompt, base64)
                } catch (e: Exception) {
                    "Generation execution halted: ${e.localizedMessage ?: e.message}"
                }

                // Treat special modes (e.g. Image Generation simulations)
                val isImgGeneration = text.trim().startsWith("/image", ignoreCase = true) || 
                                     text.trim().contains("generate an image", ignoreCase = true)

                val replyPayload = if (isImgGeneration) {
                    "Generated AI asset for input: *\"${text.removePrefix("/image").trim()}\"*"
                } else {
                    reply
                }

                // Create assistant placeholder message record inside local Room DB
                val placeholderId = repository.insertMessage(
                    ChatMessage(
                        chatId = currentChat.id,
                        role = "assistant",
                        content = "▰ Scanning core...",
                        isImageGeneration = isImgGeneration
                    )
                )

                // Split reply into words to stream token generation word-by-word with realistic delay (typing simulation)
                val words = replyPayload.split(" ")
                val typedOutput = StringBuilder()
                
                for (index in words.indices) {
                    if (!kotlin.coroutines.coroutineContext[kotlinx.coroutines.Job]?.isActive!!) {
                        break
                    }
                    typedOutput.append(words[index]).append(" ")
                    repository.updateMessageContent(placeholderId.toInt(), typedOutput.toString().trimEnd())
                    kotlinx.coroutines.delay(25) // Fast typing simulation pace
                }

                // Audit logging
                val characterLim = text.take(24)
                repository.logEvent("AI", "Generated reply for model $activeModel on prompt: \"$characterLim...\"")
            }

            // Join job and toggle loading trigger
            activeGenerationJob?.join()
            _isGenerating.value = false
        }
    }

    fun stopGeneration() {
        viewModelScope.launch {
            if (activeGenerationJob?.isActive == true) {
                activeGenerationJob?.cancel()
                val currentChat = _selectedChat.value
                if (currentChat != null) {
                    repository.insertMessage(
                        ChatMessage(
                            chatId = currentChat.id,
                            role = "assistant",
                            content = "*Generation was halted by the user.*"
                        )
                    )
                }
                repository.logEvent("AI", "LLM streaming generation interrupted by client override.")
            }
            _isGenerating.value = false
        }
    }

    fun attachSimulatedImage(label: String, base64: String) {
        _attachedImageLabel.value = label
        _attachedImageBase64.value = base64
    }

    fun clearAttachment() {
        _attachedImageLabel.value = null
        _attachedImageBase64.value = null
    }

    private fun buildSystemPrompt(model: String): String {
        return """
            ${_systemPromptPrefix.value}
            
            You are the high-fidelity native core of the AgentOS AI platform, customized and engineered for Rahul Kumar Mahto.
            Today's local date is: 2026-06-14.
            Logged-In user profile: Rahul Kumar Mahto (Founder).
            
            Operating model profile: $model.
            System Policy Filter: ${_safetyFilterLevel.value}.
            Daily API Allocation: ${_apiUsageLimit.value} calls.
            
            Please adapt your personality to reflect $model:
            - If GPT-5: Speak with highly logical precision, extreme length, and complete analytical breakdowns. Include dynamic memory tracking.
            - If Claude 3.5: Speak with rich technical insight, structured bullet points, elegant developer tone, and clean, modular formatting.
            - If Gemini: Speak in a warmly organic, creative, and highly versatile manner, summarizing complex problems into bulletproof logic.
            - If DeepSeek: Adopt a highly methodical reasoning tone. Show your "Thinking Process" enclosed inside neat blocks before presenting solutions.
            - If standard models: Be a highly elite, premium digital companion.
            
            Ensure any code output is beautifully wrapped in markdown code blocks with correct syntax tags (e.g. python, kotlin, javascript). Maintain strict formatting constraints.
        """.trimIndent()
    }


    // --- Administrator & Founder Management Functions ---

    fun promoteToAdmin(user: User) {
        viewModelScope.launch {
            repository.updateUser(user.copy(role = "admin"))
        }
    }

    fun toggleSuspension(user: User) {
        val nextStatus = if (user.status == "suspended") "active" else "suspended"
        viewModelScope.launch {
            repository.updateUser(user.copy(status = nextStatus))
            // Force Log-out of suspended user if they match current
            if (_currentUser.value?.id == user.id && nextStatus == "suspended") {
                logout()
            }
        }
    }

    fun toggleBan(user: User) {
        val nextStatus = if (user.status == "banned") "active" else "banned"
        viewModelScope.launch {
            repository.updateUser(user.copy(status = nextStatus))
            if (_currentUser.value?.id == user.id && nextStatus == "banned") {
                logout()
            }
        }
    }

    fun deleteUserAccount(user: User) {
        viewModelScope.launch {
            repository.deleteUser(user)
            if (_currentUser.value?.id == user.id) {
                logout()
            }
        }
    }

    fun demoteToUser(user: User) {
        viewModelScope.launch {
            repository.updateUser(user.copy(role = "user"))
        }
    }

    fun setMaintenanceMode(enabled: Boolean) {
        _isMaintenanceMode.value = enabled
        viewModelScope.launch {
            repository.logEvent("SECURITY_AUDIT", "Maintenance mode updated to: $enabled")
            if (enabled && _currentUser.value?.role != "founder") {
                logout()
            }
        }
    }

    fun setSafetyFilterLevel(level: String) {
        _safetyFilterLevel.value = level
        viewModelScope.launch {
            repository.logEvent("SECURITY_AUDIT", "System security level shifted to: $level")
        }
    }

    fun setAllowedRegistration(allowed: Boolean) {
        _allowedRegistration.value = allowed
        viewModelScope.launch {
            repository.logEvent("SECURITY_AUDIT", "Public user registrations allowed set to: $allowed")
        }
    }

    fun setApiUsageLimit(limit: Int) {
        _apiUsageLimit.value = limit
        viewModelScope.launch {
            repository.logEvent("SECURITY_AUDIT", "Daily API limit configured to: $limit calls per user")
        }
    }

    fun setSystemPromptPrefix(prefix: String) {
        _systemPromptPrefix.value = prefix
        viewModelScope.launch {
            repository.logEvent("SECURITY_AUDIT", "Active core system prompt customized by Founder.")
        }
    }


    // --- Memory Operations ---

    fun registerMemoryItem(key: String, value: String) {
        val user = _currentUser.value ?: return
        if (key.isBlank() || value.isBlank()) return
        viewModelScope.launch {
            repository.insertMemory(
                MemoryItem(
                    userId = user.id,
                    key = key.trim(),
                    value = value.trim()
                )
            )
        }
    }

    fun forgetMemoryItem(id: Int) {
        viewModelScope.launch {
            repository.deleteMemoryById(id)
        }
    }
}

class TaskViewModelFactory(private val repository: AgentRepository) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(TaskViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return TaskViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
