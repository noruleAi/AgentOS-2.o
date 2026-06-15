package com.example.data

import androidx.room.*

@Entity(tableName = "users", indices = [Index(value = ["email"], unique = true)])
data class User(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val username: String,
    val email: String,
    val passwordHash: String,
    val role: String, // "user", "admin", "founder"
    val status: String, // "active", "suspended", "banned"
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "chats",
    foreignKeys = [
        ForeignKey(
            entity = User::class,
            parentColumns = ["id"],
            childColumns = ["userId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["userId"])]
)
data class ChatSession(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val userId: Int,
    val title: String,
    val modelId: String,
    val isPinned: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "messages",
    foreignKeys = [
        ForeignKey(
            entity = ChatSession::class,
            parentColumns = ["id"],
            childColumns = ["chatId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["chatId"])]
)
data class ChatMessage(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val chatId: Int,
    val role: String, // "user", "assistant"
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val imageUri: String? = null,
    val isImageGeneration: Boolean = false
)

@Entity(
    tableName = "memories",
    foreignKeys = [
        ForeignKey(
            entity = User::class,
            parentColumns = ["id"],
            childColumns = ["userId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["userId"])]
)
data class MemoryItem(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val userId: Int,
    val key: String,
    val value: String,
    val timestamp: Long = System.currentTimeMillis()
)

@Entity(tableName = "app_events")
data class AppEvent(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val eventType: String, // "AUTH", "CHAT", "AI", "SECURITY_AUDIT"
    val description: String,
    val timestamp: Long = System.currentTimeMillis()
)

@Entity(tableName = "analytics")
data class AnalyticsData(
    @PrimaryKey val id: Int = 1,
    val totalUsers: Int = 0,
    val activeUsers: Int = 0,
    val totalChats: Int = 0,
    val totalMessages: Int = 0,
    val revenue: Double = 0.0
)
