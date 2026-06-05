const API_ENDPOINT = "/api/chat";

const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearChat");
const newChatButton = document.querySelector("#newChat");
const promptButtons = document.querySelectorAll(".prompt-chip");

let history = [];

const welcome = "Xin chào. Hãy hỏi về báo cáo tài chính, ví dụ doanh thu, lợi nhuận, vốn chủ sở hữu hoặc so sánh giữa các kỳ.";

function boot() {
  addMessage("assistant", welcome);
  input.focus();
}

function addMessage(role, content, options = {}) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (options.pending) row.dataset.pending = "true";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Bạn" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  row.append(avatar, bubble);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function setPending(row, content) {
  const bubble = row.querySelector(".bubble");
  bubble.textContent = content;
  row.dataset.pending = "false";
}

function resetChat() {
  history = [];
  messagesEl.innerHTML = "";
  addMessage("assistant", welcome);
  input.focus();
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

async function sendMessage(text) {
  addMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  autosize();
  sendButton.disabled = true;

  const pending = addMessage("assistant", "Đang lập kế hoạch truy xuất và đọc tài liệu...", { pending: true });
  pending.querySelector(".bubble").classList.add("typing");

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const answer = data.answer || data.message || data.response || "Không có câu trả lời từ backend.";
    pending.querySelector(".bubble").classList.remove("typing");
    setPending(pending, answer);
    history.push({ role: "assistant", content: answer });
  } catch (error) {
    const fallback = "Chưa kết nối được backend /api/chat. Backend cần nhận POST { message, history } và trả JSON có trường answer.";
    pending.querySelector(".bubble").classList.remove("typing");
    setPending(pending, `${fallback}\n\nChi tiết: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (text) sendMessage(text);
});

input.addEventListener("input", autosize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

clearButton.addEventListener("click", resetChat);
newChatButton.addEventListener("click", resetChat);
promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent.trim();
    autosize();
    input.focus();
  });
});

boot();
