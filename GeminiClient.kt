package com.example.api

import com.example.BuildConfig
import com.squareup.moshi.JsonClass
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@JsonClass(generateAdapter = true)
data class InlineData(
    val mimeType: String,
    val data: String
)

@JsonClass(generateAdapter = true)
data class Part(
    val text: String? = null,
    val inlineData: InlineData? = null
)

@JsonClass(generateAdapter = true)
data class Content(
    val parts: List<Part>
)

@JsonClass(generateAdapter = true)
data class GenerateContentRequest(
    val contents: List<Content>,
    val systemInstruction: Content? = null
)

@JsonClass(generateAdapter = true)
data class Candidate(
    val content: Content
)

@JsonClass(generateAdapter = true)
data class GenerateContentResponse(
    val candidates: List<Candidate>?
)

interface GeminiApiService {
    @POST("v1beta/models/{model}:generateContent")
    suspend fun generateContent(
        @Path("model") model: String,
        @Query("key") apiKey: String,
        @Body request: GenerateContentRequest
    ): GenerateContentResponse
}

object RetrofitClient {
    private const val BASE_URL = "https://generativelanguage.googleapis.com/"

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()

    val service: GeminiApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .build()
            .create(GeminiApiService::class.java)
    }
}

object GeminiClient {

    suspend fun generateChatResponse(
        prompt: String,
        modelId: String,
        systemPrompt: String,
        base64Image: String? = null
    ): String = withContext(Dispatchers.IO) {
        val apiKey = BuildConfig.GEMINI_API_KEY
        if (apiKey.isEmpty() || apiKey == "MY_GEMINI_API_KEY") {
            // Simulated offline backup responder if key is missing/unconfigured
            return@withContext getSimulatedResponse(prompt, modelId)
        }

        val partsList = mutableListOf<Part>()
        if (base64Image != null) {
            partsList.add(Part(inlineData = InlineData(mimeType = "image/jpeg", data = base64Image)))
        }
        partsList.add(Part(text = prompt))

        val request = GenerateContentRequest(
            contents = listOf(Content(parts = partsList)),
            systemInstruction = Content(parts = listOf(Part(text = systemPrompt)))
        )

        // Dynamic model selection maps visual options to authentic high-intelligence Gemini endpoints
        val realGeminiModelName = when (modelId.uppercase()) {
            "GPT-5" -> "gemini-1.5-pro"
            "CLAUDE" -> "gemini-1.5-pro"
            "DEEPSEEK" -> "gemini-2.0-flash"
            "GPT-4O" -> "gemini-1.5-flash"
            else -> "gemini-1.5-flash"
        }

        try {
            val response = RetrofitClient.service.generateContent(realGeminiModelName, apiKey, request)
            response.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.text 
                ?: "No response received from the agentic services."
        } catch (e: Exception) {
            "Error standardizing request: ${e.localizedMessage ?: e.message}"
        }
    }

    private fun getSimulatedResponse(prompt: String, modelId: String): String {
        return when (modelId.uppercase()) {
            "GPT-5" -> """
                [AgentOS AI Panel - GPT-5 Simulated Response]
                
                Hello Rahul Kumar Mahto! You are testing with the next-generation **GPT-5** reasoning standard. 
                Since no Gemini API key is configured in the AI Studio Secrets panel, I am operating under local offline mode.
                
                **Your input:** "$prompt"
                
                *Key Capabilities Enabled:*
                ✓ Ultra Long-Term Memory
                ✓ Multi-Agent Orchestration
                ✓ Real-time mathematical synthesis
                
                *To activate live OpenRouter / Gemini API streaming directly, please enter a valid API Key in the AI Studio Secrets panel.*
            """.trimIndent()
            
            "CLAUDE" -> """
                [AgentOS AI Panel - Claude 3.5 Sonnet Response]
                
                Greetings, Rahul. I am Claude, fine-tuned to help you coordinate, write clean architecture code, and refine system design paradigms in AgentOS.
                
                Regarding your prompt: "$prompt"
                
                Here is a brief, highly structural synthesis:
                1. **Separation of Concerns**: Keep components lightweight and decoupled.
                2. **Local Memory Isolation**: Ensure Room database caches operations cleanly.
                3. **Direct Streaming**: Standardize on SSE (Server-Sent Events) model payloads.
                
                *Configuration Note*: Operating in local simulated fallback. Enter a `GEMINI_API_KEY` to enable live cloud generation.
            """.trimIndent()

            "DEEPSEEK" -> """
                [AgentOS AI Panel - DeepSeek V3 Response]
                
                Thinking Process:
                1. Analyze Rahul Mahto's requested AgentOS setup.
                2. Identify key operational parameters in direct query: "$prompt".
                3. Formulate highly detailed, optimal solutions.
                
                DeepSeek-V3 is ready.
                
                *Solution Outline:*
                - **Asynchronous Execution**: Utilizing Coroutines and Flow pipelines prevents interface freezing.
                - **API key masking**: Ensure key reads occur solely via `BuildConfig` fields to protect integrity.
                
                *Status*: Sandbox simulation. Input your cloud key for real-time live generations.
            """.trimIndent()

            else -> """
                [AgentOS AI Panel - Assistant Mode: $modelId]
                
                Welcome to AgentOS AI! I am answering in local offline sandbox mode.
                
                **Your Prompt:** "$prompt"
                
                **To use live AI models**:
                1. Tap the **Secrets Panel** on Google AI Studio.
                2. Configure `GEMINI_API_KEY` with your valid token key.
                3. Re-run or refresh the application.
                
                Let me know what else I can construct for you, Rahul!
            """.trimIndent()
        }
    }
}
