package com.example.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface AgentDao {

    // --- Users Queries ---
    @Query("SELECT * FROM users ORDER BY id DESC")
    fun getAllUsers(): Flow<List<User>>

    @Query("SELECT * FROM users WHERE email = :email LIMIT 1")
    suspend fun getUserByEmail(email: String): User?

    @Query("SELECT * FROM users WHERE id = :id LIMIT 1")
    suspend fun getUserById(id: Int): User?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: User): Long

    @Update
    suspend fun updateUser(user: User)

    @Delete
    suspend fun deleteUser(user: User)


    // --- Chat Session Queries ---
    @Query("SELECT * FROM chats WHERE userId = :userId ORDER BY isPinned DESC, id DESC")
    fun getChatsForUser(userId: Int): Flow<List<ChatSession>>

    @Query("SELECT * FROM chats WHERE userId = :userId AND title LIKE '%' || :query || '%' ORDER BY isPinned DESC, id DESC")
    fun searchChatsForUser(userId: Int, query: String): Flow<List<ChatSession>>

    @Query("SELECT * FROM chats WHERE id = :id LIMIT 1")
    suspend fun getChatById(id: Int): ChatSession?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertChat(chat: ChatSession): Long

    @Update
    suspend fun updateChat(chat: ChatSession)

    @Query("DELETE FROM chats WHERE id = :id")
    suspend fun deleteChatById(id: Int)


    // --- Message Queries ---
    @Query("SELECT * FROM messages WHERE chatId = :chatId ORDER BY timestamp ASC")
    fun getMessagesForChat(chatId: Int): Flow<List<ChatMessage>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMessage(message: ChatMessage): Long

    @Query("UPDATE messages SET content = :newContent WHERE id = :id")
    suspend fun updateMessageContent(id: Int, newContent: String)

    @Query("DELETE FROM messages WHERE chatId = :chatId")
    suspend fun deleteMessagesForChat(chatId: Int)


    // --- Memory Queries ---
    @Query("SELECT * FROM memories WHERE userId = :userId ORDER BY timestamp DESC")
    fun getMemoriesForUser(userId: Int): Flow<List<MemoryItem>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMemory(memory: MemoryItem): Long

    @Query("DELETE FROM memories WHERE id = :id")
    suspend fun deleteMemoryById(id: Int)


    // --- Audit Log Events ---
    @Query("SELECT * FROM app_events ORDER BY timestamp DESC LIMIT 200")
    fun getAllEvents(): Flow<List<AppEvent>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEvent(event: AppEvent)


    // --- Analytics API Engine ---
    @Query("SELECT COUNT(*) FROM users")
    suspend fun getUserCount(): Int

    @Query("SELECT COUNT(*) FROM users WHERE status = 'active'")
    suspend fun getActiveUserCount(): Int

    @Query("SELECT COUNT(*) FROM chats")
    suspend fun getChatCount(): Int

    @Query("SELECT COUNT(*) FROM messages")
    suspend fun getMessageCount(): Int

    @Query("SELECT * FROM analytics WHERE id = 1 LIMIT 1")
    fun getAnalytics(): Flow<AnalyticsData?>

    @Query("SELECT * FROM analytics WHERE id = 1 LIMIT 1")
    suspend fun getAnalyticsSync(): AnalyticsData?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAnalytics(analytics: AnalyticsData)
}
