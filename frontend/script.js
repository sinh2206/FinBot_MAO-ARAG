const DEFAULT_BACKEND_PORT = "8000";

function resolveApiBase() {
  const htmlBase = document.documentElement.dataset.apiBase?.trim();
  if (htmlBase) return htmlBase.replace(/\/+$/, "");

  const override = window.APP_API_BASE?.trim?.();
  if (override) return override.replace(/\/+$/, "");

  if (window.location.port === "8080") {
    return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_BACKEND_PORT}`;
  }

  return "";
}

const API_BASE = resolveApiBase();
const API_ENDPOINT = `${API_BASE}/api/chat`;
const HEALTH_ENDPOINT = `${API_BASE}/healthz`;
const CONFIG_ENDPOINT = `${API_BASE}/api/config`;

const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearChat");
const newChatButton = document.querySelector("#newChat");
const promptButtons = document.querySelectorAll(".prompt-chip");
const topbarDescription = document.querySelector(".topbar p");
const sidebarFooter = document.querySelector(".sidebar-footer span:last-child");
const statusDot = document.querySelector(".status-dot");

let history = [];
let appConfig = null;

const welcome = "Xin chào. Hãy hỏi về báo cáo tài chính, ví dụ doanh thu, lợi nhuận, vốn chủ sở hữu hoặc so sánh giữa các kỳ.";

function boot() {
  addTextMessage("assistant", welcome, { intro: true });
  hydrateConfig();
  hydrateHealth();
  input.focus();
}

function addTextMessage(role, content, options = {}) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (options.pending) row.dataset.pending = "true";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Bạn" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = content;
  bubble.append(body);

  row.append(avatar, bubble);
  messagesEl.append(row);
  scrollToBottom();
  return row;
}

function addStructuredAssistantMessage(payload) {
  const row = document.createElement("article");
  row.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble rich-bubble";

  const answer = document.createElement("div");
  answer.className = "bubble-answer";
  answer.textContent = payload.answer || "Không có câu trả lời.";
  bubble.append(answer);

  const metaRow = document.createElement("div");
  metaRow.className = "meta-row";
  metaRow.append(
    createBadge(`Planner: Phi`, "planner"),
    createBadge(`Executor: ${payload.executor_used || "qwen"}`, payload.executor_used === "gemini" ? "fallback" : "primary"),
    createBadge(`QA: ${payload.qa_type || "n/a"}`, "qa-type")
  );
  if (payload.ticker) metaRow.append(createBadge(`Ticker: ${payload.ticker}`, "ticker"));
  bubble.append(metaRow);

  if (Array.isArray(payload.selected_sources) && payload.selected_sources.length) {
    const sourceBlock = document.createElement("div");
    sourceBlock.className = "source-block";
    const title = document.createElement("div");
    title.className = "mini-title";
    title.textContent = "Nguồn đang dùng";
    const list = document.createElement("div");
    list.className = "source-list";
    payload.selected_sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = source;
      list.append(chip);
    });
    sourceBlock.append(title, list);
    bubble.append(sourceBlock);
  }

  const diagnostics = buildDiagnostics(payload);
  if (diagnostics) bubble.append(diagnostics);

  row.append(avatar, bubble);
  messagesEl.append(row);
  scrollToBottom();
  return row;
}

function buildDiagnostics(payload) {
  const retrieval = payload.retrieval || {};
  const planner = payload.planner || {};
  const executor = payload.executor || {};
  const hasDetails =
    planner.predicted_plan ||
    planner.error ||
    retrieval.component_results?.length ||
    retrieval.context_preview ||
    executor.qwen_answer ||
    executor.gemini_answer;

  if (!hasDetails) return null;

  const details = document.createElement("details");
  details.className = "diagnostics";

  const summary = document.createElement("summary");
  summary.textContent = "Chi tiết chạy pipeline";
  details.append(summary);

  const grid = document.createElement("div");
  grid.className = "diagnostic-grid";

  const executionCard = document.createElement("section");
  executionCard.className = "diagnostic-card";
  executionCard.append(
    createMiniTitle("Tóm tắt"),
    createKeyValue("Lý do route", payload.routing_reason || "qwen_primary"),
    createKeyValue("Fallback", payload.fallback_used ? "Có" : "Không"),
    createKeyValue("Component đủ", retrieval.component_count_ok ? "Có" : "Chưa"),
    createKeyValue("Component support", formatRatio(retrieval.component_support_rate))
  );
  grid.append(executionCard);

  if (planner.predicted_plan || planner.error) {
    const plannerCard = document.createElement("section");
    plannerCard.className = "diagnostic-card";
    plannerCard.append(createMiniTitle("Planner"));
    if (planner.error) plannerCard.append(createPre(planner.error));
    if (planner.predicted_plan) plannerCard.append(createPre(JSON.stringify(planner.predicted_plan, null, 2)));
    grid.append(plannerCard);
  }

  if (retrieval.component_results?.length) {
    const retrievalCard = document.createElement("section");
    retrievalCard.className = "diagnostic-card";
    retrievalCard.append(createMiniTitle("Retrieval"));
    retrieval.component_results.forEach((item) => {
      const block = document.createElement("div");
      block.className = "component-block";
      const heading = document.createElement("div");
      heading.className = "component-heading";
      heading.textContent = `[${item.id}] ${item.query}`;
      const status = document.createElement("span");
      status.className = `component-status ${item.supported ? "ok" : "warn"}`;
      status.textContent = item.supported ? "supported" : item.supported === null ? "n/a" : "missing";
      heading.append(status);
      block.append(heading);

      (item.evidence_snippets || []).forEach((evidence) => {
        const snippet = document.createElement("div");
        snippet.className = "evidence-snippet";
        snippet.textContent = `${evidence.source}: ${evidence.snippet}`;
        block.append(snippet);
      });
      retrievalCard.append(block);
    });
    grid.append(retrievalCard);
  }

  if (retrieval.context_preview) {
    const contextCard = document.createElement("section");
    contextCard.className = "diagnostic-card";
    contextCard.append(createMiniTitle("Context tổng hợp"), createPre(retrieval.context_preview));
    grid.append(contextCard);
  }

  if (executor.qwen_answer || executor.gemini_answer) {
    const executorCard = document.createElement("section");
    executorCard.className = "diagnostic-card";
    executorCard.append(createMiniTitle("Executor"));
    if (executor.qwen_answer) executorCard.append(createKeyValue("Qwen", executor.qwen_answer));
    if (executor.gemini_answer) executorCard.append(createKeyValue("Gemini", executor.gemini_answer));
    grid.append(executorCard);
  }

  details.append(grid);
  return details;
}

function createBadge(text, variant = "") {
  const el = document.createElement("span");
  el.className = `badge ${variant}`.trim();
  el.textContent = text;
  return el;
}

function createMiniTitle(text) {
  const el = document.createElement("div");
  el.className = "mini-title";
  el.textContent = text;
  return el;
}

function createPre(text) {
  const el = document.createElement("pre");
  el.className = "diagnostic-pre";
  el.textContent = text;
  return el;
}

function createKeyValue(label, value) {
  const row = document.createElement("div");
  row.className = "kv-row";
  const key = document.createElement("span");
  key.className = "kv-key";
  key.textContent = label;
  const val = document.createElement("span");
  val.className = "kv-value";
  val.textContent = value ?? "n/a";
  row.append(key, val);
  return row;
}

function formatRatio(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(0)}%`;
}

function setPending(row, content) {
  const body = row.querySelector(".bubble-body");
  body.textContent = content;
  row.dataset.pending = "false";
}

function replacePendingWithStructured(row, payload) {
  row.remove();
  addStructuredAssistantMessage(payload);
}

function resetChat() {
  history = [];
  messagesEl.innerHTML = "";
  addTextMessage("assistant", welcome, { intro: true });
  input.focus();
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function hydrateConfig() {
  try {
    const response = await fetch(CONFIG_ENDPOINT);
    if (!response.ok) return;
    appConfig = await response.json();
    if (topbarDescription) {
      const planner = appConfig.planner || "Phi";
      const primary = appConfig.executor_primary || "Qwen";
      const fallback = appConfig.fallback_enabled ? appConfig.executor_fallback : "tắt";
      topbarDescription.textContent = `Planner: ${planner} · Executor chính: ${primary} · Gemini fallback: ${fallback}.`;
    }
  } catch (error) {
    console.error("Không tải được config backend", error);
  }
}

async function hydrateHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    if (statusDot) statusDot.classList.add("healthy");
    if (sidebarFooter) {
      const mode = health.index_summary?.retrieval_mode || "unknown";
      sidebarFooter.textContent = `API /api/chat · index ${mode}`;
    }
  } catch (error) {
    if (statusDot) statusDot.classList.add("offline");
    if (sidebarFooter) sidebarFooter.textContent = "Backend chưa sẵn sàng";
  }
}

async function sendMessage(text) {
  addTextMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  autosize();
  sendButton.disabled = true;

  const pending = addTextMessage("assistant", "Đang lập kế hoạch, truy xuất tài liệu và tổng hợp câu trả lời...", { pending: true });
  pending.querySelector(".bubble").classList.add("typing");

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    pending.querySelector(".bubble").classList.remove("typing");
    replacePendingWithStructured(pending, data);
    history.push({ role: "assistant", content: data.answer || "" });
  } catch (error) {
    pending.querySelector(".bubble").classList.remove("typing");
    setPending(
      pending,
      `Không gọi được backend /api/chat.\n\nChi tiết: ${error.message}`
    );
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
