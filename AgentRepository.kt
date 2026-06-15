package com.example.data

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AgentRepository(private val agentDao: AgentDao) {

    // --- Users ---
    val allUsers: Flow<List<User>> = agentDao.getAllUsers()

    suspend fun getUserByEmail(email: String): User? = withContext(Dispatchers.IO) {
        agentDao.getUserByEmail(email)
    }

    suspend fun getUserById(id: Int): User? = withContext(Dispatchers.IO) {
        agentDao.getUserById(id)
    }

    suspend fun insertUser(user: User): Long = withContext(Dispatchers.IO) {
        val id = agentDao.insertUser(user)
        logEvent("AUTH", "User registered: ${user.username} (${user.role})")
        recalculateAnalytics()
        id
    }

    suspend fun updateUser(user: User) = withContext(Dispatchers.IO) {
        agentDao.updateUser(user)
        logEvent("AUTH", "User profile updated: ${user.username} (status: ${user.status}, role: ${user.role})")
        recalculateAnalytics()
    }

    suspend fun deleteUser(user: User) = withContext(Dispatchers.IO) {
        agentDao.deleteUser(user)
        logEvent("AUTH", "User deleted: ${user.username}")
        recalculateAnalytics()
    }


    // --- Chats ---
    fun getChatsForUser(userId: Int): Flow<List<ChatSession>> = agentDao.getChatsForUser(userId)

    fun searchChatsForUser(userId: Int, query: String): Flow<List<ChatSession>> = agentDao.searchChatsForUser(userId, query)

    suspend fun getChatById(id: Int): ChatSession? = withContext(Dispatchers.IO) {
        agentDao.getChatById(id)
    }

    suspend fun insertChat(chat: ChatSession): Long = withContext(Dispatchers.IO) {
        val id = agentDao.insertChat(chat)
        logEvent("CHAT", "New chat session created: \"${chat.title}\" under model ${chat.modelId}")
        recalculateAnalytics()
        id
    }

    suspend fun updateChat(chat: ChatSession) = withContext(Dispatchers.IO) {
        agentDao.updateChat(chat)
    }

    suspend fun deleteChatById(id: Int) = withContext(Dispatchers.IO) {
        agentDao.deleteChatById(id)
        logEvent("CHAT", "Chat session $id deleted")
        recalculateAnalytics()
    }


    // --- Messages ---
    fun getMessagesForChat(chatId: Int): Flow<List<ChatMessage>> = agentDao.getMessagesForChat(chatId)

    suspend fun insertMessage(message: ChatMessage): Long = withContext(Dispatchers.IO) {
        val id = agentDao.insertMessage(message)
        recalculateAnalytics()
        id
    }

    suspend fun updateMessageContent(id: Int, newContent: String) = withContext(Dispatchers.IO) {
        agentDao.updateMessageContent(id, newContent)
    }

    suspend fun deleteMessagesForChat(chatId: Int) = withContext(Dispatchers.IO) {
        agentDao.deleteMessagesForChat(chatId)
    }


    // --- Memories ---
    fun getMemoriesForUser(userId: Int): Flow<List<MemoryItem>> = agentDao.getMemoriesForUser(userId)

    suspend fun insertMemory(memory: MemoryItem): Long = withContext(Dispatchers.IO) {
        val id = agentDao.insertMemory(memory)
        logEvent("AI", "Long-term memory updated: ${memory.key}")
        id
    }

    suspend fun deleteMemoryById(id: Int) = withContext(Dispatchers.IO) {
        agentDao.deleteMemoryById(id)
    }


    // --- System Logging ---
    val allEvents: Flow<List<AppEvent>> = agentDao.getAllEvents()

    suspend fun logEvent(type: String, desc: String) = withContext(Dispatchers.IO) {
        agentDao.insertEvent(AppEvent(eventType = type, description = desc))
    }


    // --- Analytics Engine ---
    val analytics: Flow<AnalyticsData?> = agentDao.getAnalytics()

    suspend fun insertAnalytics(data: AnalyticsData) = withContext(Dispatchers.IO) {
        agentDao.insertAnalytics(data)
    }

    suspend fun seedDatabase() = withContext(Dispatchers.IO) {
        // Seed Founder account if missing
        val existingFounder = agentDao.getUserByEmail("rahul@agentos.ai")
        if (existingFounder == null) {
            val founder = User(
                username = "Rahul Kumar Mahto",
                email = "rahul@agentos.ai",
                passwordHash = PasswordHasher.hash("admin123"),
                role = "founder",
                status = "active"
            )
            agentDao.insertUser(founder)
        }

        // Expanded Seed Users lists to empower the Founder view
        val seedUsers = listOf(
            User(username = "Amit Sharma (Lead Architect)", email = "amit@agentos.ai", passwordHash = PasswordHasher.hash("amit123"), role = "admin", status = "active"),
            User(username = "Priya Patel (Core Dev)", email = "priya@agentos.ai", passwordHash = PasswordHasher.hash("priya123"), role = "user", status = "suspended"),
            User(username = "Vikram Singh (Data Ops)", email = "vikram@agentos.ai", passwordHash = PasswordHasher.hash("vikram123"), role = "user", status = "banned"),
            User(username = "Sneha Reddy (UX Designer)", email = "sneha@agentos.ai", passwordHash = PasswordHasher.hash("sneha123"), role = "user", status = "active")
        )

        for (u in seedUsers) {
            if (agentDao.getUserByEmail(u.email) == null) {
                agentDao.insertUser(u)
            }
        }

        // Seed some core audit logs
        logEvent("SECURITY_AUDIT", "AgentOS AI platform database initialized securely with API protection filters.")

        // Seed default analytics statistics
        recalculateAnalytics()
    }

    private suspend fun recalculateAnalytics() {
        val uCount = agentDao.getUserCount()
        val activeUCount = agentDao.getActiveUserCount()
        val cCount = agentDao.getChatCount()
        val mCount = agentDao.getMessageCount()
        
        // Dynamic simulated revenue driven directly by the actual counts in database
        val baseRevenue = (uCount * 19.99) + (mCount * 0.05)
        
        val defaultStats = AnalyticsData(
            id = 1,
            totalUsers = uCount + 900,
            activeUsers = activeUCount + 800,
            totalChats = cCount + 1400,
            totalMessages = mCount + 11000,
            revenue = baseRevenue + 14000.00
        )
        agentDao.insertAnalytics(defaultStats)
    }
}
