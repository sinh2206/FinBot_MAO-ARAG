// ===== DOM ELEMENTS =====
const welcomeScreen = document.getElementById('welcome-screen');
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const historyList = document.getElementById('chat-history-list');
const modal = document.getElementById('sub-modal');
const openModalBtn = document.getElementById('open-sub-modal');
const closeModalBtn = document.querySelector('.close-modal');
const confirmSubBtn = document.getElementById('confirm-sub');
const uploadBtn = document.getElementById('upload-btn'); // chÆ°a dÃ¹ng

// ===== STATE =====
let isWaiting = false; // Ä‘ang chá» pháº£n há»“i tá»« server
let currentMessages = []; // lÆ°u táº¡m tin nháº¯n hiá»‡n táº¡i (cÃ³ thá»ƒ dÃ¹ng Ä‘á»ƒ restore history)
let sessionId = Date.now().toString(); // táº¡m thá»i dÃ¹ng timestamp
const userId = 'anonymous';

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    // Load lá»‹ch sá»­ chat (náº¿u cÃ³)
    loadHistory();

    // Focus vÃ o input
    userInput.focus();

    // Auto-resize textarea
    userInput.addEventListener('input', autoResize);

    // Gá»­i khi nháº¥n Enter (khÃ´ng Shift)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Xá»­ lÃ½ nÃºt gá»­i
    sendBtn.addEventListener('click', sendMessage);

    // NÃºt new chat
    newChatBtn.addEventListener('click', resetChat);

    // Modal
    openModalBtn.addEventListener('click', () => {
        modal.classList.remove('hidden');
    });

    closeModalBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });

    confirmSubBtn.addEventListener('click', () => {
        // Giáº£ láº­p Ä‘Äƒng kÃ½, cÃ³ thá»ƒ thÃ´ng bÃ¡o
        alert('Cáº£m Æ¡n báº¡n Ä‘Ã£ Ä‘Äƒng kÃ½!');
        modal.classList.add('hidden');
    });

    // Xá»­ lÃ½ nÃºt upload (táº¡m thá»i khÃ´ng lÃ m)
    uploadBtn.addEventListener('click', () => {
        alert('Chá»©c nÄƒng Ä‘Ã­nh kÃ¨m Ä‘ang phÃ¡t triá»ƒn.');
    });
});

// ===== AUTO RESIZE TEXTAREA =====
function autoResize() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    // Giá»›i háº¡n tá»‘i Ä‘a 200px (Ä‘Ã£ cÃ³ trong CSS)
    if (this.scrollHeight > 200) {
        this.style.height = '200px';
        this.style.overflowY = 'auto';
    } else {
        this.style.overflowY = 'hidden';
    }

    // Enable/disable send button dá»±a trÃªn ná»™i dung
    if (this.value.trim() === '') {
        sendBtn.disabled = true;
    } else {
        sendBtn.disabled = false;
    }
}

// ===== SEND MESSAGE =====
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isWaiting) return;

    // áº¨n welcome screen, hiá»‡n chat container
    welcomeScreen.classList.add('hidden');
    chatContainer.classList.remove('hidden');

    // ThÃªm tin nháº¯n user vÃ o chat
    addMessageToChat('user', message);

    // XÃ³a input vÃ  reset chiá»u cao
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Hiá»ƒn thá»‹ typing indicator
    const typingId = showTypingIndicator();

    isWaiting = true;

    try {
        // Gá»i API
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                user_id: userId,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        // data: { answer, cost, workflow, latency, token_usage }

        // XÃ³a typing indicator
        removeTypingIndicator(typingId);

        // Render response
        renderBotResponse(data.answer, data.chart_path || null);

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', 'âŒ Xin lá»—i, Ä‘Ã£ xáº£y ra lá»—i. Vui lÃ²ng thá»­ láº¡i sau.');
    } finally {
        isWaiting = false;
    }
}

// ===== RENDER BOT RESPONSE =====
function renderBotResponse(markdownText, chartPath) {
    // Parse markdown thÃ nh HTML
    const htmlContent = marked.parse(markdownText);

    // Táº¡o container cho bot message
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-container bot';

    // Avatar
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';
    messageDiv.appendChild(avatarDiv);

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = htmlContent;
    messageDiv.appendChild(contentDiv);

    // Náº¿u cÃ³ chartPath, thÃªm áº£nh
    if (chartPath) {
        const img = document.createElement('img');
        img.src = chartPath;
        img.alt = 'Biá»ƒu Ä‘á»“';
        img.style.maxWidth = '100%';
        contentDiv.appendChild(img);
    }

    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    // LÆ°u vÃ o currentMessages (cÃ³ thá»ƒ dÃ¹ng Ä‘á»ƒ sau nÃ y)
    currentMessages.push({ role: 'bot', content: markdownText });
}

// ===== ADD USER MESSAGE =====
function addMessageToChat(role, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-container ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    if (role === 'user') {
        avatarDiv.innerHTML = '<i class="fa-regular fa-user"></i>';
    } else {
        avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';
    }
    messageDiv.appendChild(avatarDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text; // user message khÃ´ng markdown
    messageDiv.appendChild(contentDiv);

    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    // LÆ°u vÃ o currentMessages
    currentMessages.push({ role, content: text });
}

// ===== TYPING INDICATOR =====
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message-container bot';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';
    typingDiv.appendChild(avatarDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    typingDiv.appendChild(contentDiv);

    chatContainer.appendChild(typingDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const typing = document.getElementById(id);
    if (typing) typing.remove();
}

// ===== SCROLL TO BOTTOM =====
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ===== RESET CHAT (New Chat) =====
function resetChat() {
    // XÃ³a toÃ n bá»™ tin nháº¯n trong chat container
    chatContainer.innerHTML = '';
    // Hiá»‡n láº¡i welcome screen
    welcomeScreen.classList.remove('hidden');
    chatContainer.classList.add('hidden');
    // Clear current messages
    currentMessages = [];
    // Focus input
    userInput.focus();
}

// ===== LOAD HISTORY =====
async function loadHistory() {
    try {
        const response = await fetch(`/history/${encodeURIComponent(userId)}`);
        if (!response.ok) throw new Error('Failed to load history');
        const data = await response.json();
        // Render vÃ o historyList
        const items = Array.isArray(data?.history) ? data.history : [];
        if (items.length > 0) {
            historyList.innerHTML = '';
            items.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.dataset.id = item.id;
                historyItem.innerHTML = `
                    <i class="fa-regular fa-message"></i>
                    <span>${item.query || 'Cuá»™c trÃ² chuyá»‡n'}</span>
                `;
                historyItem.addEventListener('click', () => loadConversation(item.id));
                historyList.appendChild(historyItem);
            });
        } else {
            // Hiá»ƒn thá»‹ placeholder
            historyList.innerHTML = '<div class="history-item" style="justify-content:center;">ChÆ°a cÃ³ lá»‹ch sá»­</div>';
        }
    } catch (error) {
        console.warn('Could not load history:', error);
        // Fallback: hiá»ƒn thá»‹ máº«u
        historyList.innerHTML = `
            <div class="history-item"><i class="fa-regular fa-message"></i> <span>PhÃ¢n tÃ­ch FPT</span></div>
            <div class="history-item"><i class="fa-regular fa-message"></i> <span>So sÃ¡nh HPG vÃ  HSG</span></div>
        `;
    }
}

// ===== LOAD CONVERSATION (khi click vÃ o history) =====
function loadConversation(convId) {
    // á»ž Ä‘Ã¢y cÃ³ thá»ƒ gá»i API láº¥y chi tiáº¿t cuá»™c trÃ² chuyá»‡n vÃ  render láº¡i
    // NhÆ°ng Ä‘á»ƒ Ä‘Æ¡n giáº£n, ta chá»‰ thÃ´ng bÃ¡o
    alert(`TÃ­nh nÄƒng Ä‘ang phÃ¡t triá»ƒn. Conversation ID: ${convId}`);
}

// ===== SET INPUT Tá»ª SUGGESTION CARD =====
// HÃ m setInput Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trong HTML global, nhÆ°ng ta cÃ³ thá»ƒ override náº¿u cáº§n
window.setInput = function(text) {
    userInput.value = text;
    userInput.focus();
    userInput.dispatchEvent(new Event('input'));
};
