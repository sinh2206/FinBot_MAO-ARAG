/**
 * script.js - Xử lý toàn bộ logic chat frontend
 * Bao gồm: gửi tin nhắn, hiệu ứng loading, gọi API, render Markdown + biểu đồ.
 */

// DOM elements
const messagesContainer = document.getElementById('messagesContainer');
const queryInput = document.getElementById('queryInput');
const sendButton = document.getElementById('sendButton');

// Biến trạng thái
let isLoading = false;          // Đang gửi tin nhắn?
let currentMessageId = null;    // ID của tin nhắn tạm (loading)
let loadingInterval = null;     // Interval để thay đổi trạng thái loading
const loadingSteps = [
    "⏳ Đang phân tích câu hỏi...",
    "📊 Đang lấy dữ liệu giá từ VnStock...",
    "📑 Đang đọc báo cáo tài chính Q3...",
    "📈 Đang vẽ biểu đồ...",
    "🤔 Đang tổng hợp thông tin...",
    "✍️ Đang viết báo cáo..."
];

/**
 * Tự động điều chỉnh chiều cao của textarea
 */
queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

/**
 * Gửi tin nhắn khi nhấn Enter (không Shift)
 */
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/**
 * Gửi tin nhắn khi click nút gửi
 */
sendButton.addEventListener('click', sendMessage);

/**
 * Hàm gửi tin nhắn chính
 */
async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Hiển thị tin nhắn của user
    addUserMessage(query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Bắt đầu trạng thái loading
    startLoading();

    try {
        // Gọi API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, user_id: 'anonymous' })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // data format: { answer: "...", chart_url: "/charts/...", processing_time: "...", cost: "..." }
        
        // Kết thúc loading và hiển thị kết quả
        finishLoading(data);
    } catch (error) {
        console.error('Lỗi khi gọi API:', error);
        finishLoadingWithError('Xin lỗi, đã xảy ra lỗi kết nối. Vui lòng thử lại sau.');
    }
}

/**
 * Thêm tin nhắn của user vào container
 */
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';

    messageDiv.innerHTML = `
        <div class="avatar"><i class="fas fa-user"></i></div>
        <div class="message-content">
            <p>${escapeHtml(text)}</p>
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Thêm tin nhắn của assistant (AI)
 */
function addAssistantMessage(markdownContent, chartUrl = null, processingTime = '', cost = '') {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    // Render markdown
    const htmlContent = marked.parse(markdownContent);

    let chartHtml = '';
    if (chartUrl) {
        chartHtml = `<img src="${chartUrl}" alt="Biểu đồ chứng khoán" loading="lazy">`;
    }

    let footerHtml = '';
    if (processingTime || cost) {
        footerHtml = `<div class="message-footer">
            ${processingTime ? `<span>⏱️ ${processingTime}</span>` : ''}
            ${cost ? `<span>💰 ${cost}</span>` : ''}
        </div>`;
    }

    messageDiv.innerHTML = `
        <div class="avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            ${htmlContent}
            ${chartHtml}
            ${footerHtml}
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Bắt đầu hiệu ứng loading: tạo tin nhắn tạm và chạy interval thay đổi text
 */
function startLoading() {
    isLoading = true;
    sendButton.disabled = true;

    // Tạo tin nhắn loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            <p class="loading-step"><i class="fas fa-spinner fa-pulse"></i> ${loadingSteps[0]}</p>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();

    let stepIndex = 0;
    loadingInterval = setInterval(() => {
        stepIndex = (stepIndex + 1) % loadingSteps.length;
        const stepElement = document.querySelector('#loading-message .loading-step');
        if (stepElement) {
            stepElement.innerHTML = `<i class="fas fa-spinner fa-pulse"></i> ${loadingSteps[stepIndex]}`;
        }
    }, 2000); // Đổi mỗi 2 giây
}

/**
 * Kết thúc loading thành công, xóa tin nhắn tạm và hiển thị kết quả
 */
function finishLoading(data) {
    // Dọn dẹp interval và xóa tin nhắn loading
    clearInterval(loadingInterval);
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) loadingMsg.remove();

    // Thêm tin nhắn assistant với dữ liệu từ API
    addAssistantMessage(
        data.answer,
        data.chart_url,
        data.processing_time,
        data.cost
    );

    isLoading = false;
    sendButton.disabled = false;
}

/**
 * Kết thúc loading với lỗi
 */
function finishLoadingWithError(errorMessage) {
    clearInterval(loadingInterval);
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) loadingMsg.remove();

    // Thêm tin nhắn lỗi
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message assistant';
    errorDiv.innerHTML = `
        <div class="avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content" style="color: #ff6b6b;">
            <p><i class="fas fa-exclamation-triangle"></i> ${escapeHtml(errorMessage)}</p>
        </div>
    `;
    messagesContainer.appendChild(errorDiv);
    scrollToBottom();

    isLoading = false;
    sendButton.disabled = false;
}

/**
 * Helper: cuộn xuống cuối cùng
 */
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Helper: escape HTML để tránh XSS (cho user message)
 */
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Load lịch sử chat (có thể gọi API /api/history/{user_id} để lấy)
async function loadHistory() {
    try {
        const response = await fetch('/api/history/anonymous?limit=20');
        if (response.ok) {
            const data = await response.json();
            const historyList = document.getElementById('historyList');
            // Xóa các item cũ (giữ lại header)
            historyList.innerHTML = '';
            data.history.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.innerHTML = `<i class="fas fa-message"></i><span>${escapeHtml(item.query.substring(0, 30))}...</span>`;
                historyItem.addEventListener('click', () => {
                    // Có thể set lại câu hỏi vào input hoặc tự động gửi
                    queryInput.value = item.query;
                });
                historyList.appendChild(historyItem);
            });
        }
    } catch (e) {
        console.warn('Không thể load lịch sử:', e);
    }
}

// Gọi load lịch sử khi trang được tải
window.addEventListener('load', () => {
    loadHistory();
});