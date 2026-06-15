package com.example

import android.app.Application
import com.example.data.AgentDatabase
import com.example.data.AgentRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class TaskApplication : Application() {
    lateinit var database: AgentDatabase
        private set

    lateinit var repository: AgentRepository
        private set

    override fun onCreate() {
        super.onCreate()
        database = AgentDatabase.getDatabase(this)
        repository = AgentRepository(database.agentDao())

        // Seed default dataset asynchronously on startup
        CoroutineScope(Dispatchers.IO).launch {
            repository.seedDatabase()
        }
    }
}
