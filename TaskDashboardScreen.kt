package com.example.ui

import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.data.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TaskDashboardScreen(
    viewModel: TaskViewModel,
    modifier: Modifier = Modifier
) {
    val currentUser by viewModel.currentUser.collectAsState()
    val authError by viewModel.authError.collectAsState()
    val authSuccess by viewModel.authSuccess.collectAsState()
    val context = LocalContext.current

    // Trigger Toast alerts for Auth events
    LaunchedEffect(authError) {
        authError?.let {
            Toast.makeText(context, it, Toast.LENGTH_LONG).show()
        }
    }

    LaunchedEffect(authSuccess) {
        authSuccess?.let {
            Toast.makeText(context, it, Toast.LENGTH_SHORT).show()
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        if (currentUser == null) {
            // Sleek portal login gate
            AgentAuthPortal(viewModel = viewModel)
        } else {
            // Main adaptive Workspace
            AgentOSWorkspace(viewModel = viewModel, user = currentUser!!)
        }
    }
}

// ==========================================
// 1. AUTHENTICATION / ACCESS GATEWAY
// ==========================================

@Composable
fun AgentAuthPortal(viewModel: TaskViewModel) {
    var isRegisterMode by remember { mutableStateOf(false) }
    var username by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPasswordVisible by remember { mutableStateOf(false) }

    val gradientBrush = Brush.verticalGradient(
        colors = listOf(
            Color(0xFF0F172A), // Dark Slate
            Color(0xFF1E293B), // Soft Slate
            Color(0xFF064E3B)  // Dark Emerald Glow
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradientBrush),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .widthIn(max = 450.dp)
                .background(Color(0xE61E293B), RoundedCornerShape(24.dp))
                .border(BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.3f)), RoundedCornerShape(24.dp))
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Elegant glowing logo representing AgentOS
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(Brush.radialGradient(listOf(Color(0xFF34D399), Color(0xFF059669))))
                    .wrapContentSize(Alignment.Center)
            ) {
                Icon(
                    imageVector = Icons.Rounded.Terminal,
                    contentDescription = "AgentOS Icon",
                    tint = Color(0xFF0F172A),
                    modifier = Modifier.size(36.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "AgentOS AI Workspace",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                fontFamily = FontFamily.SansSerif,
                textAlign = TextAlign.Center
            )

            Text(
                text = "Secure Containerized Client Gatev1.0",
                fontSize = 12.sp,
                color = Color(0xFF34D399),
                fontWeight = FontWeight.Medium,
                letterSpacing = 1.sp,
                modifier = Modifier.padding(top = 4.dp, bottom = 24.dp)
            )

            // Dynamic Form layout
            if (isRegisterMode) {
                OutlinedTextField(
                    value = username,
                    onValueChange = { username = it },
                    label = { Text("Display Username", color = Color(0xFF94A3B8)) },
                    placeholder = { Text("e.g. Rahul Mahto") },
                    colors = authTextFieldColors(),
                    singleLine = true,
                    leadingIcon = { Icon(Icons.Rounded.Person, contentDescription = null, tint = Color(0xFF34D399)) },
                    modifier = Modifier.fillMaxWidth().testTag("auth_username_field")
                )
                Spacer(modifier = Modifier.height(12.dp))
            }

            OutlinedTextField(
                value = email,
                onValueChange = { email = it },
                label = { Text("Workspace Email", color = Color(0xFF94A3B8)) },
                placeholder = { Text("e.g. rahul@agentos.ai") },
                colors = authTextFieldColors(),
                singleLine = true,
                leadingIcon = { Icon(Icons.Rounded.Mail, contentDescription = null, tint = Color(0xFF34D399)) },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email, imeAction = ImeAction.Next),
                modifier = Modifier.fillMaxWidth().testTag("auth_email_field")
            )

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = password,
                onValueChange = { password = it },
                label = { Text("Access Password", color = Color(0xFF94A3B8)) },
                placeholder = { Text("Password credentials") },
                colors = authTextFieldColors(),
                singleLine = true,
                leadingIcon = { Icon(Icons.Rounded.Lock, contentDescription = null, tint = Color(0xFF34D399)) },
                trailingIcon = {
                    IconButton(onClick = { isPasswordVisible = !isPasswordVisible }) {
                        Icon(
                            imageVector = if (isPasswordVisible) Icons.Rounded.Visibility else Icons.Rounded.VisibilityOff,
                            contentDescription = "Toggle password visibility",
                            tint = Color(0xFF34D399)
                        )
                    }
                },
                visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                modifier = Modifier.fillMaxWidth().testTag("auth_password_field")
            )

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = {
                    if (isRegisterMode) {
                        viewModel.register(username, email, password)
                    } else {
                        viewModel.login(email, password)
                    }
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF10B981),
                    contentColor = Color(0xFF0F172A)
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
                    .testTag("auth_submit_button")
            ) {
                Text(
                    text = if (isRegisterMode) "Create Account Profile" else "Access Container",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = if (isRegisterMode) "Have credentials?" else "Register container?",
                    color = Color(0xFF94A3B8),
                    fontSize = 12.sp,
                    modifier = Modifier
                        .clickable { isRegisterMode = !isRegisterMode }
                        .padding(4.dp)
                )

                if (!isRegisterMode) {
                    Text(
                        text = "Reset credential ID",
                        color = Color(0xFF34D399),
                        fontSize = 12.sp,
                        modifier = Modifier
                            .clickable { viewModel.triggerPasswordReset(email) }
                            .padding(4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Divider(color = Color(0xFF334155), thickness = 1.dp)
            Spacer(modifier = Modifier.height(16.dp))

            // Sandbox quick tests utilities
            Text(
                text = "Sandbox Platform Testing:",
                fontSize = 11.sp,
                color = Color(0xFF64748B),
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.align(Alignment.Start)
            )

            Spacer(modifier = Modifier.height(6.dp))

            Button(
                onClick = { viewModel.guestLogin() },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF3F3F46),
                    contentColor = Color.White
                ),
                modifier = Modifier.fillMaxWidth().height(42.dp),
                shape = RoundedCornerShape(8.dp)
            ) {
                Text("Enter as Guest Operator", fontSize = 13.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun authTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = Color(0xFF10B981),
    unfocusedBorderColor = Color(0xFF334155),
    focusedLabelColor = Color(0xFF34D399),
    unfocusedLabelColor = Color(0xFF94A3B8),
    focusedTextColor = Color.White,
    unfocusedTextColor = Color.White,
    focusedContainerColor = Color(0xFF1E293B),
    unfocusedContainerColor = Color(0xFF1E293B)
)


// ==========================================
// 2. MAIN ADAPTIVE WORKSPACE
// ==========================================

@OptIn(ExperimentalAnimationApi::class, ExperimentalMaterial3Api::class)
@Composable
fun AgentOSWorkspace(
    viewModel: TaskViewModel,
    user: User
) {
    val coroutineScope = rememberCoroutineScope()
    var isHistoryDrawerOpen by remember { mutableStateOf(false) }
    var isFounderPanelOpen by remember { mutableStateOf(false) }
    var isSettingsOpen by remember { mutableStateOf(false) }

    val activeChat by viewModel.selectedChat.collectAsState()
    val allChatsList by viewModel.allChats.collectAsState()
    val chatQuery by viewModel.searchQuery.collectAsState()

    // Render side-by-side on wide screens using BoxWithConstraints
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val useTwinPanes = maxWidth >= 800.dp

        Row(modifier = Modifier.fillMaxSize()) {
            // Adaptive left column drawer for wider displays
            if (useTwinPanes) {
                Box(
                    modifier = Modifier
                        .width(280.dp)
                        .fillMaxHeight()
                        .background(Color(0xFF0F172A))
                        .border(BorderStroke(1.dp, Color(0xFF1E293B)))
                ) {
                    WorkspaceSidebar(
                        viewModel = viewModel,
                        chats = allChatsList,
                        query = chatQuery,
                        activeChat = activeChat,
                        user = user,
                        onOpenSettings = { isSettingsOpen = true },
                        onOpenFounderPanel = { isFounderPanelOpen = true }
                    )
                }
            }

            // Central Chat & Input column
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .background(Color(0xFF0F172A))
            ) {
                CentralChatWorkspace(
                    viewModel = viewModel,
                    activeChat = activeChat,
                    user = user,
                    hasSidebarToggle = !useTwinPanes,
                    onToggleSidebar = { isHistoryDrawerOpen = true },
                    onOpenSettings = { isSettingsOpen = true },
                    onOpenFounderPanel = { isFounderPanelOpen = true }
                )
            }
        }

        // Drop-In Drawer for narrower viewports
        if (!useTwinPanes && isHistoryDrawerOpen) {
            Dialog(
                onDismissRequest = { isHistoryDrawerOpen = false }
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.85f)
                        .fillMaxHeight(0.9f)
                        .background(Color(0xFF0F172A), RoundedCornerShape(16.dp))
                        .border(BorderStroke(1.dp, Color(0xFF1E293B)), RoundedCornerShape(16.dp))
                        .padding(16.dp)
                ) {
                    Column(modifier = Modifier.fillMaxSize()) {
                        IconButton(
                            onClick = { isHistoryDrawerOpen = false },
                            modifier = Modifier.align(Alignment.End)
                        ) {
                            Icon(Icons.Rounded.Close, contentDescription = "Close history drawer", tint = Color.White)
                        }
                        WorkspaceSidebar(
                            viewModel = viewModel,
                            chats = allChatsList,
                            query = chatQuery,
                            activeChat = activeChat,
                            user = user,
                            onOpenSettings = {
                                isHistoryDrawerOpen = false
                                isSettingsOpen = true
                            },
                            onOpenFounderPanel = {
                                isHistoryDrawerOpen = false
                                isFounderPanelOpen = true
                            }
                        )
                    }
                }
            }
        }

        // Dialog Modals for Subcomponents
        if (isFounderPanelOpen) {
            FounderDashboardModal(viewModel = viewModel, onDismiss = { isFounderPanelOpen = false })
        }

        if (isSettingsOpen) {
            SettingsModal(viewModel = viewModel, onDismiss = { isSettingsOpen = false })
        }
    }
}


// ==========================================
// 3. WORKSPACE SIDEBAR (CHATS, PINS, LOGOUT)
// ==========================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WorkspaceSidebar(
    viewModel: TaskViewModel,
    chats: List<ChatSession>,
    query: String,
    activeChat: ChatSession?,
    user: User,
    onOpenSettings: () -> Unit,
    onOpenFounderPanel: () -> Unit
) {
    val pinnedChats = chats.filter { it.isPinned }
    val remainingChats = chats.filter { !it.isPinned }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // App identity tag
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(
                imageVector = Icons.Rounded.Storage,
                contentDescription = null,
                tint = Color(0xFF10B981),
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = "AgentOS AI Workspace",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Create Chat button
        Button(
            onClick = { viewModel.createChatSession() },
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF1E293B),
                contentColor = Color(0xFF10B981)
            ),
            shape = RoundedCornerShape(10.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(40.dp)
                .testTag("create_chat_button"),
            border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.4f))
        ) {
            Icon(Icons.Rounded.Add, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("Launch New Agent", fontSize = 13.sp, fontWeight = FontWeight.Bold)
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Sidebar Chat Filter Search Field
        OutlinedTextField(
            value = query,
            onValueChange = { viewModel.setSearchQuery(it) },
            placeholder = { Text("Search sessions...", color = Color(0xFF64748B), fontSize = 12.sp) },
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF10B981),
                unfocusedBorderColor = Color(0xFF1E293B),
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedContainerColor = Color(0xFF0F172A),
                unfocusedContainerColor = Color(0xFF0F172A)
            ),
            singleLine = true,
            leadingIcon = { Icon(Icons.Rounded.Search, contentDescription = null, tint = Color(0xFF64748B), modifier = Modifier.size(16.dp)) },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        // List Scroll Container
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            if (pinnedChats.isNotEmpty()) {
                item {
                    Text(
                        text = "PINNED ENGINES",
                        color = Color(0xFF34D399),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        modifier = Modifier.padding(vertical = 4.dp, horizontal = 8.dp)
                    )
                }

                items(pinnedChats, key = { "pinned_${it.id}" }) { chat ->
                    ChatSessionItemRow(chat = chat, activeChat = activeChat, viewModel = viewModel)
                }

                item { Spacer(modifier = Modifier.height(12.dp)) }
            }

            item {
                Text(
                    text = "RUNNING CONTEXTS",
                    color = Color(0xFF64748B),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp,
                    modifier = Modifier.padding(vertical = 4.dp, horizontal = 8.dp)
                )
            }

            if (remainingChats.isEmpty()) {
                item {
                    Text(
                        text = "No previous historic setups.",
                        color = Color(0xFF475569),
                        fontSize = 12.sp,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                    )
                }
            } else {
                items(remainingChats, key = { "chat_${it.id}" }) { chat ->
                    ChatSessionItemRow(chat = chat, activeChat = activeChat, viewModel = viewModel)
                }
            }
        }

        Divider(color = Color(0xFF1E293B), thickness = 1.dp, modifier = Modifier.padding(vertical = 12.dp))

        // User Status identity Bar & Option list
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Identity badges
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E293B), RoundedCornerShape(12.dp))
                    .padding(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF10B981).copy(alpha = 0.2f))
                        .wrapContentSize(Alignment.Center)
                ) {
                    Text(
                        text = user.username.take(1).uppercase(),
                        color = Color(0xFF10B981),
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }

                Spacer(modifier = Modifier.width(8.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = user.username,
                        color = Color.White,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = user.role.uppercase(),
                        color = if (user.role == "founder") Color(0xFFEF4444) else Color(0xFF34D399),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                IconButton(onClick = { viewModel.logout() }) {
                    Icon(Icons.Rounded.Logout, contentDescription = "Log out", tint = Color(0xFFEF4444), modifier = Modifier.size(16.dp))
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Option switches
            if (user.role == "founder" || user.role == "admin") {
                Button(
                    onClick = onOpenFounderPanel,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF451A03),
                        contentColor = Color(0xFFFDBA74)
                    ),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth().height(36.dp),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                ) {
                    Icon(Icons.Rounded.DashboardCustomize, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Founder Console Panel", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }

            Button(
                onClick = onOpenSettings,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF1E293B),
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth().height(36.dp),
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
            ) {
                Icon(Icons.Rounded.Tune, contentDescription = null, modifier = Modifier.size(14.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("App Setup & Long-term Memory", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun ChatSessionItemRow(
    chat: ChatSession,
    activeChat: ChatSession?,
    viewModel: TaskViewModel
) {
    val isActive = activeChat?.id == chat.id
    var isEditing by remember { mutableStateOf(false) }
    var editTitleText by remember { mutableStateOf(chat.title) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (isActive) Color(0xFF1E293B) else Color.Transparent)
            .clickable { if (!isActive) viewModel.selectChat(chat) }
            .padding(vertical = 4.dp, horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = if (chat.isPinned) Icons.Rounded.Bookmark else Icons.Rounded.ChatBubbleOutline,
            contentDescription = null,
            tint = if (isActive) Color(0xFF10B981) else Color(0xFF64748B),
            modifier = Modifier.size(16.dp)
        )

        Spacer(modifier = Modifier.width(8.dp))

        if (isEditing) {
            BasicTextField(
                value = editTitleText,
                onValueChange = { editTitleText = it },
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = Color.White, fontSize = 13.sp),
                modifier = Modifier
                    .weight(1f)
                    .background(Color(0xFF0F172A), RoundedCornerShape(4.dp))
                    .padding(4.dp),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = {
                    isEditing = false
                    viewModel.renameChat(chat, editTitleText)
                })
            )
        } else {
            Text(
                text = chat.title,
                color = if (isActive) Color.White else Color(0xFF94A3B8),
                fontSize = 13.sp,
                fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
        }

        if (isActive) {
            IconButton(
                onClick = { 
                    if (isEditing) {
                        viewModel.renameChat(chat, editTitleText)
                    } else {
                        editTitleText = chat.title
                    }
                    isEditing = !isEditing 
                },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = if (isEditing) Icons.Rounded.Check else Icons.Rounded.Edit,
                    contentDescription = "Rename Chat",
                    tint = Color(0xFF10B981),
                    modifier = Modifier.size(14.dp)
                )
            }

            IconButton(
                onClick = { viewModel.pinChat(chat) },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = if (chat.isPinned) Icons.Rounded.BookmarkRemove else Icons.Rounded.Bookmark,
                    contentDescription = "Pin Chat",
                    tint = Color(0xFF34D399),
                    modifier = Modifier.size(14.dp)
                )
            }

            IconButton(
                onClick = { viewModel.deleteChat(chat) },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = Icons.Rounded.Delete,
                    contentDescription = "Delete Chat",
                    tint = Color(0xFFEF4444),
                    modifier = Modifier.size(14.dp)
                )
            }
        }
    }
}

// Simple layout wrapper to fix Compose basic textfield imports cleanly
@Composable
fun BasicTextField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    textStyle: androidx.compose.ui.text.TextStyle = androidx.compose.ui.text.TextStyle.Default,
    keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
    keyboardActions: KeyboardActions = KeyboardActions.Default
) {
    androidx.compose.foundation.text.BasicTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier,
        textStyle = textStyle,
        keyboardOptions = keyboardOptions,
        keyboardActions = keyboardActions,
        cursorBrush = Brush.verticalGradient(listOf(Color.White, Color.White))
    )
}


// ==========================================
// 4. CENTRAL CHAT CONTAINER CORE
// ==========================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CentralChatWorkspace(
    viewModel: TaskViewModel,
    activeChat: ChatSession?,
    user: User,
    hasSidebarToggle: Boolean,
    onToggleSidebar: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenFounderPanel: () -> Unit
) {
    val messages by viewModel.chatMessages.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val activeModel by viewModel.selectedModel.collectAsState()

    var isModelSelectorOpen by remember { mutableStateOf(false) }
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()
    val keyboardController = LocalSoftwareKeyboardController.current

    val modelsList = listOf("GPT-4o", "GPT-5", "Claude 3.5", "Gemini 1.5", "DeepSeek V3", "Llama 3", "Mistral")

    // Automatic down-scrolling on replies list
    LaunchedEffect(messages.size, isGenerating) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Core control header
        TopAppBar(
            title = {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = activeChat?.title ?: "AgentOS Central Unit",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        fontSize = 15.sp,
                        modifier = Modifier.weight(1f, fill = false)
                    )

                    // Model drop-down switcher selector
                    Box {
                        Text(
                            text = activeModel,
                            color = Color(0xFF10B981),
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            modifier = Modifier
                                .background(Color(0xFF1E293B), RoundedCornerShape(12.dp))
                                .border(BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.5f)), RoundedCornerShape(12.dp))
                                .clickable { isModelSelectorOpen = true }
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        )

                        DropdownMenu(
                            expanded = isModelSelectorOpen,
                            onDismissRequest = { isModelSelectorOpen = false },
                            modifier = Modifier.background(Color(0xFF1E293B))
                        ) {
                            modelsList.forEach { model ->
                                DropdownMenuItem(
                                    text = { Text(model, color = Color.White, fontSize = 13.sp) },
                                    onClick = {
                                        viewModel.changeModelSelection(model)
                                        isModelSelectorOpen = false
                                    }
                                )
                            }
                        }
                    }
                }
            },
            navigationIcon = {
                if (hasSidebarToggle) {
                    IconButton(onClick = onToggleSidebar) {
                        Icon(Icons.Rounded.Menu, contentDescription = "Drawer menu", tint = Color.White)
                    }
                } else {
                    Icon(
                        imageVector = Icons.Rounded.OfflineBolt,
                        contentDescription = "Active connection status",
                        tint = Color(0xFF10B981),
                        modifier = Modifier.padding(start = 12.dp)
                    )
                }
            },
            actions = {
                if (user.role == "founder" || user.role == "admin") {
                    IconButton(onClick = onOpenFounderPanel) {
                        Icon(Icons.Rounded.Analytics, contentDescription = "Open Admin Panel", tint = Color(0xFFF59E0B))
                    }
                }
                IconButton(onClick = onOpenSettings) {
                    Icon(Icons.Rounded.Settings, contentDescription = "Open Settings", tint = Color.White)
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = Color(0xFF0F172A)
            )
        )

        Divider(color = Color(0xFF1E293B), thickness = 1.dp)

        // Chat timeline space
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            if (activeChat == null) {
                // Landing suggestions dashboard (like ChatGPT)
                ChatDashboardEmptyState(
                    user = user,
                    onStartPrompt = { prompt ->
                        viewModel.createChatSession(customTitle = prompt.take(24))
                        viewModel.sendMessage(prompt)
                    }
                )
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 14.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    contentPadding = PaddingValues(vertical = 16.dp)
                ) {
                    items(messages, key = { it.id }) { message ->
                        ChatMessageItemCard(message = message)
                    }

                    if (isGenerating) {
                        item {
                            TypingWaitingIndicatorCard(modelLabel = activeChat.modelId)
                        }
                    }
                }
            }
        }

        Divider(color = Color(0xFF1E293B), thickness = 1.dp)

        // Active attachments bar view
        val currentAttachmentLabel by viewModel.attachedImageLabel.collectAsState()
        if (currentAttachmentLabel != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E293B))
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Rounded.Image, contentDescription = null, tint = Color(0xFF34D399), modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Attached for Analysis: **$currentAttachmentLabel**",
                    color = Color.White,
                    fontSize = 11.sp,
                    modifier = Modifier.weight(1f)
                )
                IconButton(onClick = { viewModel.clearAttachment() }, modifier = Modifier.size(24.dp)) {
                    Icon(Icons.Rounded.Close, contentDescription = "Clear attachment", tint = Color.Red, modifier = Modifier.size(14.dp))
                }
            }
            Divider(color = Color(0xFF1E293B), thickness = 1.dp)
        }

        // Dictation microphone ripple modal mock
        var isMicrophoneModalOpen by remember { mutableStateOf(false) }
        var isFileAttachModalOpen by remember { mutableStateOf(false) }

        if (isMicrophoneModalOpen) {
            DictationMockDialog(
                onDismiss = { isMicrophoneModalOpen = false },
                onCapturedText = { dictation ->
                    inputText = (inputText + " " + dictation).trim()
                    isMicrophoneModalOpen = false
                }
            )
        }

        if (isFileAttachModalOpen) {
            FileAttachmentMockDialog(
                onDismiss = { isFileAttachModalOpen = false },
                onFileSelected = { label, base64 ->
                    viewModel.attachSimulatedImage(label, base64)
                    isFileAttachModalOpen = false
                }
            )
        }

        // Prompt input area
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (activeChat != null) {
                // Attach asset button
                IconButton(
                    onClick = { isFileAttachModalOpen = true },
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color(0xFF1E293B), CircleShape)
                ) {
                    Icon(Icons.Rounded.AttachFile, contentDescription = "Attach file object", tint = Color(0xFF94A3B8))
                }

                // Vocal recorder simulation button
                IconButton(
                    onClick = { isMicrophoneModalOpen = true },
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color(0xFF1E293B), CircleShape)
                ) {
                    Icon(Icons.Rounded.Mic, contentDescription = "Microphone dynamic dictation", tint = Color(0xFF94A3B8))
                }
            }

            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                placeholder = {
                    Text(
                        text = if (activeChat == null) "Authorize or create a new session above first..." else "Command AgentOS, Rahul...",
                        color = Color(0xFF64748B),
                        fontSize = 13.sp
                    )
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF10B981),
                    unfocusedBorderColor = Color(0xFF1E293B),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedContainerColor = Color(0xFF1E293B),
                    unfocusedContainerColor = Color(0xFF1E293B)
                ),
                shape = RoundedCornerShape(20.dp),
                enabled = activeChat != null,
                maxLines = 4,
                singleLine = false,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = {
                    if (inputText.isNotBlank() && !isGenerating) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                        keyboardController?.hide()
                    }
                }),
                modifier = Modifier
                    .weight(1f)
                    .testTag("chat_input_field")
            )

            if (activeChat != null) {
                if (isGenerating) {
                    IconButton(
                        onClick = { viewModel.stopGeneration() },
                        modifier = Modifier
                            .size(40.dp)
                            .background(Color(0xFFEF4444), CircleShape)
                    ) {
                        Icon(Icons.Rounded.Stop, contentDescription = "Halt streaming", tint = Color.White)
                    }
                } else {
                    IconButton(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                viewModel.sendMessage(inputText)
                                inputText = ""
                                keyboardController?.hide()
                            }
                        },
                        enabled = inputText.isNotBlank(),
                        modifier = Modifier
                            .size(40.dp)
                            .background(
                                if (inputText.isNotBlank()) Color(0xFF10B981) else Color(0xFF334155),
                                CircleShape
                            )
                            .testTag("chat_send_button")
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.ArrowUpward,
                            contentDescription = "Transmit message prompt",
                            tint = if (inputText.isNotBlank()) Color(0xFF0F172A) else Color(0xFF94A3B8)
                        )
                    }
                }
            }
        }
    }
}


// ==========================================
// 5. CHAT MESSAGE CARDS RENDERING SYSTEM
// ==========================================

@Composable
fun ChatMessageItemCard(message: ChatMessage) {
    val isAssistant = message.role == "assistant"
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = if (isAssistant) Arrangement.Start else Arrangement.End
    ) {
        if (isAssistant) {
            Box(
                modifier = Modifier
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF10B981).copy(alpha = 0.2f))
                    .wrapContentSize(Alignment.Center)
            ) {
                Icon(
                    Icons.Rounded.Memory,
                    contentDescription = null,
                    tint = Color(0xFF34D399),
                    modifier = Modifier.size(16.dp)
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
        }

        Column(
            modifier = Modifier
                .fillMaxWidth(0.85f),
            horizontalAlignment = if (isAssistant) Alignment.Start else Alignment.End
        ) {
            // Visual label
            if (message.imageUri != null) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .background(Color(0xFF1E293B), RoundedCornerShape(topStart = 8.dp, topEnd = 8.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Icon(Icons.Rounded.CameraAlt, contentDescription = null, tint = Color(0xFF34D399), modifier = Modifier.size(12.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(text = "Visual File Analyzed: ${message.imageUri}", color = Color.White, fontSize = 9.sp, fontWeight = FontWeight.Bold)
                }
            }

            // Normal Card / AI generated canvas
            if (message.isImageGeneration) {
                ImageGenerationRendererCard(promptText = message.content)
            } else {
                Box(
                    modifier = Modifier
                        .background(
                            color = if (isAssistant) Color(0xFF1E293B) else Color(0xFF10B981),
                            shape = RoundedCornerShape(
                                topStart = if (isAssistant) 0.dp else 16.dp,
                                topEnd = if (isAssistant) 16.dp else 0.dp,
                                bottomStart = 16.dp,
                                bottomEnd = 16.dp
                            )
                        )
                        .padding(12.dp)
                ) {
                    Column {
                        // Scan for Code highlights ( ``` )
                        val content = message.content
                        if (content.contains("```")) {
                            MarkdownWithCodeBlocks(content = content, onCopyCode = { code ->
                                clipboardManager.setText(AnnotatedString(code))
                                Toast.makeText(context, "Code copied to clipboard!", Toast.LENGTH_SHORT).show()
                            })
                        } else {
                            Text(
                                text = content,
                                color = if (isAssistant) Color.White else Color(0xFF0F172A),
                                fontSize = 14.sp,
                                lineHeight = 20.sp
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(2.dp))

            // Clipboard triggers
            if (isAssistant && !message.isImageGeneration) {
                Text(
                    text = "Copy text",
                    fontSize = 11.sp,
                    color = Color(0xFF64748B),
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .clickable {
                            clipboardManager.setText(AnnotatedString(message.content))
                            Toast.makeText(context, "Copied content!", Toast.LENGTH_SHORT).show()
                        }
                        .padding(horizontal = 4.dp, vertical = 2.dp)
                )
            }
        }

        if (!isAssistant) {
            Spacer(modifier = Modifier.width(8.dp))
            Box(
                modifier = Modifier
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF10B981))
                    .wrapContentSize(Alignment.Center)
            ) {
                Icon(
                    Icons.Rounded.Person,
                    contentDescription = null,
                    tint = Color(0xFF0F172A),
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

@Composable
fun MarkdownWithCodeBlocks(
    content: String,
    onCopyCode: (String) -> Unit
) {
    val segments = content.split("```")
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        segments.forEachIndexed { index, segment ->
            if (index % 2 == 1) { // It's a code block
                val lines = segment.lines()
                val language = lines.firstOrNull()?.trim() ?: ""
                val codeOnly = lines.drop(1).joinToString("\n").trim()

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                        .border(BorderStroke(1.dp, Color(0xFF334155)), RoundedCornerShape(8.dp))
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF1E293B))
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = language.uppercase().ifBlank { "CODE" },
                            color = Color(0xFF34D399),
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.clickable { onCopyCode(codeOnly) }
                        ) {
                            Icon(Icons.Rounded.ContentCopy, contentDescription = null, tint = Color.White, modifier = Modifier.size(12.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Copy", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                    Text(
                        text = codeOnly,
                        color = Color(0xFF34D399),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            } else { // Standard text segments
                if (segment.isNotBlank()) {
                    Text(
                        text = segment.trim(),
                        color = Color.White,
                        fontSize = 14.sp,
                        lineHeight = 20.sp
                    )
                }
            }
        }
    }
}

@Composable
fun ImageGenerationRendererCard(promptText: String) {
    Column(
        modifier = Modifier
            .width(260.dp)
            .background(Color(0xFF1E293B), RoundedCornerShape(16.dp))
            .border(BorderStroke(1.dp, Color(0xFF10B981)), RoundedCornerShape(16.dp))
            .padding(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.Brush, contentDescription = null, tint = Color(0xFF10B981), modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("AgentOS Artwork Studio", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Soft visual canvas container to mimic dynamic imagery
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(160.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            Color(0xFF047857), // Deep emerald
                            Color(0xFF4C1D95), // Deep Purple
                            Color(0xFF0F172A)  // Slate
                        )
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = Color(0xFF34D399), modifier = Modifier.size(48.dp))
                Spacer(modifier = Modifier.height(8.dp))
                Text("AI Canvas Export v1K", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 11.sp)
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = promptText,
            color = Color(0xFF94A3B8),
            fontSize = 11.sp,
            lineHeight = 15.sp,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
fun TypingWaitingIndicatorCard(modelLabel: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.Start,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(28.dp)
                .clip(CircleShape)
                .background(Color(0xFF34D399).copy(alpha = 0.2f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Rounded.Memory, contentDescription = null, tint = Color(0xFF34D399), modifier = Modifier.size(16.dp))
        }

        Spacer(modifier = Modifier.width(8.dp))

        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text("$modelLabel formulating", color = Color(0xFF94A3B8), fontSize = 12.sp, fontWeight = FontWeight.Medium)
                
                // Infinite dots animations simulated
                var dotsCount by remember { mutableStateOf(1) }
                LaunchedEffect(Unit) {
                    while (true) {
                        delay(400)
                        dotsCount = (dotsCount % 3) + 1
                    }
                }
                Text(".".repeat(dotsCount), color = Color(0xFF10B981), fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}


// ==========================================
// 6. CHAT EMPTY DASHBOARD SUGGESTIONS
// ==========================================

@Composable
fun ChatDashboardEmptyState(
    user: User,
    onStartPrompt: (String) -> Unit
) {
    val suggestions = listOf(
        "Suggest FastAPI endpoints logic in Python for JWT validations on Railway",
        "Explain PostgreSQL cascading deletes in Room SQLite databases",
        "Compose code checking user status active/suspended in Android ViewModel",
        "Design a glowing neon card component in Jetpack Compose",
        "Generate a futuristic tech logo idea using geometric canvas vector rules"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(18.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Rounded.Lightbulb,
            contentDescription = null,
            tint = Color(0xFFF59E0B),
            modifier = Modifier.size(56.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Initiate AgentOS Cloud Model Pipeline",
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp,
            textAlign = TextAlign.Center
        )

        Text(
            text = "Welcome, ${user.username}! Choose a preset query parameter to begin orchestrating sandbox tasks.",
            color = Color(0xFF94A3B8),
            fontSize = 12.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .padding(top = 4.dp, bottom = 24.dp)
        )

        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 500.dp)
        ) {
            suggestions.forEach { prompt ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onStartPrompt(prompt) },
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, Color(0xFF334155))
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Rounded.KeyboardArrowRight,
                            contentDescription = null,
                            tint = Color(0xFF10B981),
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = prompt,
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}


// ==========================================
// 7. DICTATION & ATTACHMENT MOCK MODALS
// ==========================================

@Composable
fun DictationMockDialog(
    onDismiss: () -> Unit,
    onCapturedText: (String) -> Unit
) {
    var rippleScale by remember { mutableStateOf(1f) }
    var secondsCount by remember { mutableStateOf(3) }

    LaunchedEffect(Unit) {
        while (secondsCount > 0) {
            delay(1000)
            secondsCount--
        }
        onCapturedText("Rahul requested optimizing PostgreSQL table schemas and launching FastAPI Docker containers.")
    }

    LaunchedEffect(Unit) {
        while (true) {
            rippleScale = 1.3f
            delay(500)
            rippleScale = 1f
            delay(500)
        }
    }

    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.85f)
                .background(Color(0xFF1E293B), RoundedCornerShape(16.dp))
                .border(BorderStroke(1.dp, Color(0xFF10B981)), RoundedCornerShape(16.dp))
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Vocal Transcriber Dictation Active", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 15.sp)
            Spacer(modifier = Modifier.height(20.dp))

            // Animated pulsing microphone circle
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF10B981).copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF10B981)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Rounded.Mic, contentDescription = null, tint = Color(0xFF0F172A), modifier = Modifier.size(28.dp))
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Text("Listening... (Auto-transcribing in $secondsCount seconds)", color = Color(0xFF94A3B8), fontSize = 11.sp)
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = onDismiss, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444))) {
                Text("Cancel", color = Color.White, fontSize = 11.sp)
            }
        }
    }
}

@Composable
fun FileAttachmentMockDialog(
    onDismiss: () -> Unit,
    onFileSelected: (String, String) -> Unit
) {
    // Elegant base64 mocks to feed visual processing
    val filesList = listOf(
        Pair("Whiteboard Wireframe diagram.png", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="),
        Pair("FastAPI JWT source snippet.py", "bXlfZmFzdF9hcGlfZGF0YV9zdHJ1Y3R1cmVfXzE="),
        Pair("Railway dockerfile deployment.yaml", "ZG9ja2VyZmlsZQ=="),
        Pair("Selfie Profile photo.jpg", "cGhvdG9fYmFzZTY0"),
        Pair("PostgreSQL SQL table schema.sql", "c3FsX3NjaGVtYQ==")
    )

    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .background(Color(0xFF1E293B), RoundedCornerShape(16.dp))
                .border(BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.5f)), RoundedCornerShape(16.dp))
                .padding(20.dp)
        ) {
            Text("Select Sandbox Object for AI Vision", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Spacer(modifier = Modifier.height(12.dp))

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                filesList.forEach { file ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                            .clickable { onFileSelected(file.first, file.second) }
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Rounded.InsertDriveFile, contentDescription = null, tint = Color(0xFF10B981), modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(file.first, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155)),
                modifier = Modifier.align(Alignment.End)
            ) {
                Text("Dismiss", color = Color.White)
            }
        }
    }
}


// ==========================================
// 8. FOUNDER DASHBOARD & LOGS VIEWER
// ==========================================

@Composable
fun FounderDashboardModal(
    viewModel: TaskViewModel,
    onDismiss: () -> Unit
) {
    val analytics by viewModel.analytics.collectAsState()
    val usersList by viewModel.allUsersList.collectAsState()
    val eventsList by viewModel.systemEvents.collectAsState()

    var activeTab by remember { mutableStateOf("METRICS") } // METRICS, USERS, CONFIGURE, LOGS

    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.9f)
                .background(Color(0xFF0F172A), RoundedCornerShape(20.dp))
                .border(BorderStroke(1.dp, Color(0xFFF59E0B)), RoundedCornerShape(20.dp))
                .padding(18.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.AdminPanelSettings, contentDescription = null, tint = Color(0xFFF59E0B), modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("AgentOS Founder Console", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Rounded.Close, contentDescription = null, tint = Color.Red)
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Navigation tabs - METRICS, USERS, CONFIGURE, LOGS
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                listOf("METRICS", "USERS", "CONFIGURE", "LOGS").forEach { tab ->
                    Button(
                        onClick = { activeTab = tab },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (activeTab == tab) Color(0xFF1E293B) else Color.Transparent,
                            contentColor = if (activeTab == tab) Color(0xFFF59E0B) else Color(0xFF94A3B8)
                        ),
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text(tab, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Body depending on selected tab
            Box(modifier = Modifier.weight(1f)) {
                when (activeTab) {
                    "METRICS" -> AnalyticsDashboardView(analytics)
                    "USERS" -> UsersManagementView(usersList, viewModel)
                    "CONFIGURE" -> SystemConfigurationView(viewModel)
                    "LOGS" -> SystemEventsLogView(eventsList)
                }
            }
        }
    }
}

@Composable
fun AnalyticsDashboardView(metrics: AnalyticsData?) {
    val items = listOf(
        Triple("TOTAL REGISTRATIONS", "${metrics?.totalUsers ?: 942}", Color(0xFF34D399)),
        Triple("ACTIVE OPERATORS", "${metrics?.activeUsers ?: 841}", Color(0xFF60A5FA)),
        Triple("TOTAL CHAT MESSAGES", "${metrics?.totalMessages ?: 11488}", Color(0xFFA78BFA)),
        Triple("PLATFORM REVENUE", "$${metrics?.revenue ?: 14592.50}", Color(0xFFFBBF24))
    )

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Container metrics derived from SQLite analytics engine:", color = Color(0xFF94A3B8), fontSize = 12.sp)

        items.forEach { metric ->
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(metric.first, color = Color(0xFF64748B), fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(metric.second, color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                    }

                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(metric.third.copy(alpha = 0.2f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Rounded.TrendingUp, contentDescription = null, tint = metric.third, modifier = Modifier.size(18.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun UsersManagementView(users: List<User>, viewModel: TaskViewModel) {
    var searchQuery by remember { mutableStateOf("") }
    
    val filteredUsers = remember(users, searchQuery) {
        if (searchQuery.isBlank()) {
            users
        } else {
            users.filter { 
                it.username.contains(searchQuery, ignoreCase = true) || 
                it.email.contains(searchQuery, ignoreCase = true)
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Search operator field
        OutlinedTextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            placeholder = { Text("Search operators by email or name...", color = Color(0xFF64748B), fontSize = 12.sp) },
            textColorText = Color.White,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 12.dp)
        )

        if (filteredUsers.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("No operators located matching query criteria.", color = Color(0xFF64748B), fontSize = 12.sp)
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.weight(1f)
            ) {
                items(filteredUsers, key = { it.id }) { u ->
                    Card(
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, Color(0xFF334155))
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Box(
                                        modifier = Modifier
                                            .size(32.dp)
                                            .clip(CircleShape)
                                            .background(
                                                if (u.role == "founder") Color(0xFFEF4444).copy(alpha = 0.2f)
                                                else if (u.role == "admin") Color(0xFF3B82F6).copy(alpha = 0.2f)
                                                else Color(0xFF10B981).copy(alpha = 0.2f)
                                            )
                                            .wrapContentSize(Alignment.Center)
                                    ) {
                                        Text(
                                            text = u.username.take(1).uppercase(),
                                            color = if (u.role == "founder") Color(0xFFEF4444) else if (u.role == "admin") Color(0xFF3B82F6) else Color(0xFF10B981),
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 13.sp
                                        )
                                    }

                                    Spacer(modifier = Modifier.width(10.dp))

                                    Column {
                                        Text(u.username, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                        Text(u.email, color = Color(0xFF94A3B8), fontSize = 11.sp)
                                    }
                                }

                                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                    // Role Badge
                                    Text(
                                        text = u.role.uppercase(),
                                        color = if (u.role == "founder") Color(0xFFEF4444) else if (u.role == "admin") Color(0xFF3B82F6) else Color(0xFF10B981),
                                        fontSize = 8.sp,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .background(Color(0xFF0F172A), RoundedCornerShape(4.dp))
                                            .padding(horizontal = 6.dp, vertical = 2.dp)
                                    )

                                    // Status Badge
                                    Text(
                                        text = u.status.uppercase(),
                                        color = when(u.status.lowercase()) {
                                            "active" -> Color(0xFF10B981)
                                            "suspended" -> Color(0xFFFBBF24)
                                            else -> Color(0xFFEF4444)
                                        },
                                        fontSize = 8.sp,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .background(Color(0xFF0F172A), RoundedCornerShape(4.dp))
                                            .padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }

                            if (u.role != "founder") {
                                Spacer(modifier = Modifier.height(10.dp))
                                Divider(color = Color(0xFF334155), thickness = 0.5.dp)
                                Spacer(modifier = Modifier.height(8.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    // Demote/Promote toggle
                                    if (u.role == "admin") {
                                        Button(
                                            onClick = { viewModel.demoteToUser(u) },
                                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155), contentColor = Color(0xFF94A3B8)),
                                            shape = RoundedCornerShape(6.dp),
                                            contentPadding = PaddingValues(horizontal = 6.dp),
                                            modifier = Modifier.height(28.dp)
                                        ) {
                                            Icon(Icons.Rounded.RemoveModerator, contentDescription = null, modifier = Modifier.size(12.dp))
                                            Spacer(modifier = Modifier.width(3.dp))
                                            Text("Demote", fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                        }
                                    } else {
                                        Button(
                                            onClick = { viewModel.promoteToAdmin(u) },
                                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF161E2E), contentColor = Color(0xFF3B82F6)),
                                            shape = RoundedCornerShape(6.dp),
                                            contentPadding = PaddingValues(horizontal = 6.dp),
                                            modifier = Modifier.height(28.dp),
                                            border = BorderStroke(1.dp, Color(0xFF3B82F6).copy(alpha = 0.3f))
                                        ) {
                                            Icon(Icons.Rounded.AddModerator, contentDescription = null, modifier = Modifier.size(12.dp))
                                            Spacer(modifier = Modifier.width(3.dp))
                                            Text("Admin", fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                        }
                                    }

                                    // Suspend/Lift suspension
                                    val isSuspended = u.status == "suspended"
                                    Button(
                                        onClick = { viewModel.toggleSuspension(u) },
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = if (isSuspended) Color(0xFF78350F) else Color(0xFF1E293B),
                                            contentColor = if (isSuspended) Color(0xFFFBBF24) else Color(0xFFF59E0B)
                                        ),
                                        shape = RoundedCornerShape(6.dp),
                                        contentPadding = PaddingValues(horizontal = 6.dp),
                                        modifier = Modifier.height(28.dp),
                                        border = BorderStroke(1.dp, Color(0xFFF59E0B).copy(alpha = 0.3f))
                                    ) {
                                        Icon(
                                            imageVector = if (isSuspended) Icons.Rounded.Block else Icons.Rounded.Lock,
                                            contentDescription = null,
                                            modifier = Modifier.size(12.dp)
                                        )
                                        Spacer(modifier = Modifier.width(3.dp))
                                        Text(if (isSuspended) "Lift Susp" else "Suspend", fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                    }

                                    // Ban/Unban toggle
                                    val isBanned = u.status == "banned"
                                    Button(
                                        onClick = { viewModel.toggleBan(u) },
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = if (isBanned) Color(0xFF7F1D1D) else Color(0xFF1E293B),
                                            contentColor = if (isBanned) Color(0xFFFCA5A5) else Color(0xFFEF4444)
                                        ),
                                        shape = RoundedCornerShape(6.dp),
                                        contentPadding = PaddingValues(horizontal = 6.dp),
                                        modifier = Modifier.height(28.dp),
                                        border = BorderStroke(1.dp, Color(0xFFEF4444).copy(alpha = 0.3f))
                                    ) {
                                        Icon(
                                            imageVector = if (isBanned) Icons.Rounded.LockOpen else Icons.Rounded.Close,
                                            contentDescription = null,
                                            modifier = Modifier.size(12.dp)
                                        )
                                        Spacer(modifier = Modifier.width(3.dp))
                                        Text(if (isBanned) "Unban" else "Ban", fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                    }

                                    Spacer(modifier = Modifier.weight(1f))

                                    // Permanent Delete control
                                    IconButton(
                                        onClick = { viewModel.deleteUserAccount(u) },
                                        modifier = Modifier
                                            .size(28.dp)
                                            .background(Color(0xFF7F1D1D).copy(alpha = 0.1f), RoundedCornerShape(6.dp))
                                    ) {
                                        Icon(
                                            imageVector = Icons.Rounded.Delete,
                                            contentDescription = "Delete Operator Record",
                                            tint = Color(0xFFEF4444),
                                            modifier = Modifier.size(14.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SystemConfigurationView(viewModel: TaskViewModel) {
    val isMaintenance by viewModel.isMaintenanceMode.collectAsState()
    val safetyLevel by viewModel.safetyFilterLevel.collectAsState()
    val allowedReg by viewModel.allowedRegistration.collectAsState()
    val apiLimit by viewModel.apiUsageLimit.collectAsState()
    val promptPrefix by viewModel.systemPromptPrefix.collectAsState()

    var customPrefixText by remember { mutableStateOf(promptPrefix) }

    LazyColumn(
        verticalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(vertical = 4.dp)
    ) {
        item {
            Text(
                text = "Secure System Configuration Console",
                color = Color(0xFFF59E0B),
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
            Text(
                text = "Decretal operational overrides of AgentOS. Live updates apply across instances immediately.",
                color = Color(0xFF94A3B8),
                fontSize = 11.sp,
                lineHeight = 15.sp,
                modifier = Modifier.padding(top = 2.dp, bottom = 4.dp)
            )
        }

        // Maintenance Mode Card
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(if (isMaintenance) Color(0xFFEF4444).copy(alpha = 0.2f) else Color(0xFF10B981).copy(alpha = 0.2f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Construction,
                            contentDescription = null,
                            tint = if (isMaintenance) Color(0xFFEF4444) else Color(0xFF10B981),
                            modifier = Modifier.size(18.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(12.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Maintenance Safe Mode",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                        Text(
                            text = if (isMaintenance) "Active. Access restricted to Founder." else "Inactive. Normal operator access.",
                            color = Color(0xFF94A3B8),
                            fontSize = 10.sp
                        )
                    }

                    Switch(
                        checked = isMaintenance,
                        onCheckedChange = { viewModel.setMaintenanceMode(it) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = Color(0xFFEF4444)
                        )
                    )
                }
            }
        }

        // Allowed Registrations Card
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(if (allowedReg) Color(0xFF10B981).copy(alpha = 0.2f) else Color(0xFFEF4444).copy(alpha = 0.2f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.AppRegistration,
                            contentDescription = null,
                            tint = if (allowedReg) Color(0xFF10B981) else Color(0xFFEF4444),
                            modifier = Modifier.size(18.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(12.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Allowed Guest Registrations",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                        Text(
                            text = if (allowedReg) "Registration Pipeline OPEN." else "Registration Pipeline CLOSED.",
                            color = Color(0xFF94A3B8),
                            fontSize = 10.sp
                        )
                    }

                    Switch(
                        checked = allowedReg,
                        onCheckedChange = { viewModel.setAllowedRegistration(it) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = Color(0xFF10B981)
                        )
                    )
                }
            }
        }

        // API Limit Selector Card
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF3B82F6).copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Rounded.Speed,
                                contentDescription = null,
                                tint = Color(0xFF3B82F6),
                                modifier = Modifier.size(18.dp)
                            )
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Column {
                            Text(
                                text = "Daily Operator API Call Allocations",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp
                            )
                            Text(
                                text = "Inquiries authorized per operator per cycle.",
                                color = Color(0xFF94A3B8),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Button(
                            onClick = { if (apiLimit > 50) viewModel.setApiUsageLimit(apiLimit - 50) },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155)),
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = PaddingValues(0.dp),
                            modifier = Modifier.size(width = 48.dp, height = 32.dp)
                        ) {
                            Text("-50", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }

                        Text(
                            text = "$apiLimit calls/operator",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.weight(1f)
                        )

                        Button(
                            onClick = { if (apiLimit < 1000) viewModel.setApiUsageLimit(apiLimit + 50) },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155)),
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = PaddingValues(0.dp),
                            modifier = Modifier.size(width = 48.dp, height = 32.dp)
                        ) {
                            Text("+50", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // Safety Policy Filter Levels Card
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF8B5CF6).copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Rounded.Security,
                                contentDescription = null,
                                tint = Color(0xFF8B5CF6),
                                modifier = Modifier.size(18.dp)
                            )
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Column {
                            Text(
                                text = "Policy Safety Isolation Level",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp
                            )
                            Text(
                                text = "Current level: $safetyLevel",
                                color = Color(0xFF94A3B8),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf("Standard Safety", "Enhanced Strict", "Dev Override").forEach { level ->
                            val isSelected = safetyLevel == level
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (isSelected) Color(0xFF8B5CF6) else Color(0xFF161E2E))
                                    .border(BorderStroke(1.dp, if (isSelected) Color.Transparent else Color(0xFF334155)), RoundedCornerShape(8.dp))
                                    .clickable { viewModel.setSafetyFilterLevel(level) }
                                    .padding(vertical = 6.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = level,
                                    color = if (isSelected) Color.White else Color(0xFF94A3B8),
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }
            }
        }

        // Core Prefix Prompt Customizer
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, Color(0xFF334155))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .background(Color(0xFF10B981).copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Rounded.Terminal,
                                contentDescription = null,
                                tint = Color(0xFF10B981),
                                modifier = Modifier.size(18.dp)
                            )
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Column {
                            Text(
                                text = "Executive Prompt Compilation Prefix",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp
                            )
                            Text(
                                text = "Base instructions injected globally into system frameworks.",
                                color = Color(0xFF94A3B8),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    androidx.compose.foundation.text.BasicTextField(
                        value = customPrefixText,
                        onValueChange = { customPrefixText = it },
                        textStyle = androidx.compose.ui.text.TextStyle(color = Color.White, fontSize = 11.sp, fontFamily = FontFamily.Monospace),
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0F172A), RoundedCornerShape(8.dp))
                            .border(BorderStroke(1.dp, Color(0xFF334155)), RoundedCornerShape(8.dp))
                            .padding(8.dp)
                            .heightIn(min = 60.dp)
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    Button(
                        onClick = { viewModel.setSystemPromptPrefix(customPrefixText) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981), contentColor = Color(0xFF0F172A)),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().height(36.dp)
                    ) {
                        Icon(Icons.Rounded.Save, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Hot-apply Custom Prompt Prefix", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

@Composable
fun SystemEventsLogView(events: List<AppEvent>) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items(events, key = { it.id }) { ev ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E293B), RoundedCornerShape(4.dp))
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "[${ev.eventType}]",
                    color = when (ev.eventType.uppercase()) {
                        "SECURITY_AUDIT" -> Color(0xFFEF4444)
                        "AUTH" -> Color(0xFF3B82F6)
                        "AI" -> Color(0xFF10B981)
                        else -> Color.Cyan
                    },
                    fontWeight = FontWeight.Bold,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.width(100.dp)
                )

                Spacer(modifier = Modifier.width(8.dp))

                Text(
                    text = ev.description,
                    color = Color.White,
                    fontSize = 11.sp,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}


// ==========================================
// 9. SETTINGS & LONG-TERM MEMORY DIALOG
// ==========================================

@Composable
fun SettingsModal(
    viewModel: TaskViewModel,
    onDismiss: () -> Unit
) {
    val memories by viewModel.memories.collectAsState()
    var newKey by remember { mutableStateOf("") }
    var newValue by remember { mutableStateOf("") }

    val context = LocalContext.current

    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.85f)
                .background(Color(0xFF0F172A), RoundedCornerShape(20.dp))
                .border(BorderStroke(1.dp, Color(0xFF10B981)), RoundedCornerShape(20.dp))
                .padding(18.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.Memory, contentDescription = null, tint = Color(0xFF10B981))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("AgentOS Long-term Memory", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 15.sp)
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Rounded.Close, contentDescription = null, tint = Color.Red)
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            Text(
                text = "These points are permanently injected into the system prompt across model launches:",
                color = Color(0xFF94A3B8),
                fontSize = 11.sp,
                lineHeight = 15.sp
            )

            Spacer(modifier = Modifier.height(14.dp))

            // Memory input
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                OutlinedTextField(
                    value = newKey,
                    onValueChange = { newKey = it },
                    placeholder = { Text("Topic key", fontSize = 11.sp) },
                    textColorText = Color.White,
                    modifier = Modifier.weight(1f)
                )

                OutlinedTextField(
                    value = newValue,
                    onValueChange = { newValue = it },
                    placeholder = { Text("Topic value details", fontSize = 11.sp) },
                    textColorText = Color.White,
                    modifier = Modifier.weight(1.5f)
                )

                Button(
                    onClick = {
                        if (newKey.isNotBlank() && newValue.isNotBlank()) {
                            viewModel.registerMemoryItem(newKey, newValue)
                            newKey = ""
                            newValue = ""
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
                    contentPadding = PaddingValues(0.dp),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.size(height = 48.dp, width = 48.dp)
                ) {
                    Icon(Icons.Rounded.Check, contentDescription = "Add memory", tint = Color(0xFF0F172A))
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Memory list
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (memories.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(24.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("Memory container empty.", color = Color(0xFF475569), fontSize = 12.sp)
                        }
                    }
                } else {
                    items(memories, key = { it.id }) { item ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Color(0xFF1E293B), RoundedCornerShape(8.dp))
                                .padding(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(item.key.uppercase(), color = Color(0xFF34D399), fontWeight = FontWeight.Bold, fontSize = 10.sp)
                                Text(item.value, color = Color.White, fontSize = 12.sp)
                            }
                            IconButton(onClick = { viewModel.forgetMemoryItem(item.id) }, modifier = Modifier.size(28.dp)) {
                                Icon(Icons.Rounded.Delete, contentDescription = "Forgot memory", tint = Color.Red, modifier = Modifier.size(16.dp))
                            }
                        }
                    }
                }
            }
        }
    }
}

// Simple OutlinedTextField to keep custom bindings clean and compiled
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OutlinedTextField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: @Composable () -> Unit,
    textColorText: Color,
    modifier: Modifier = Modifier
) {
    TextField(
        value = value,
        onValueChange = onValueChange,
        placeholder = placeholder,
        colors = TextFieldDefaults.colors(
            focusedContainerColor = Color(0xFF1E293B),
            unfocusedContainerColor = Color(0xFF1E293B),
            focusedTextColor = textColorText,
            unfocusedTextColor = textColorText,
            focusedIndicatorColor = Color(0xFF10B981)
        ),
        shape = RoundedCornerShape(8.dp),
        modifier = modifier.height(48.dp)
    )
}
