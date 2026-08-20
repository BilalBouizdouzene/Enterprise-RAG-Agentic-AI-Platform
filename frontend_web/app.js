const state = {
  sessionId: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `session-${Date.now()}`,
  metrics: null,
  documents: [],
  messages: [],
  health: null,
  recognition: null,
  listening: false,
};

const API_KEY_STORAGE_KEY = "northstar.api_key";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function getApiKey() {
  try {
    return window.localStorage.getItem(API_KEY_STORAGE_KEY) || "";
  } catch (error) {
    return "";
  }
}

function setApiKey(value) {
  try {
    if (value) window.localStorage.setItem(API_KEY_STORAGE_KEY, value);
    else window.localStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch (error) {
    // Private browsing modes can disable localStorage; the current request can still proceed.
  }
}

function openApiKeyModal(message = "") {
  const modal = $("#api-key-modal");
  if (!modal) return;
  $("#api-key-input").value = getApiKey();
  $("#api-key-status").textContent = message;
  modal.classList.add("open");
  window.setTimeout(() => $("#api-key-input").focus(), 0);
}

function closeApiKeyModal() {
  $("#api-key-modal")?.classList.remove("open");
}

async function api(path, options = {}) {
  const request = { ...options };
  const authHeaders = {};
  const apiKey = getApiKey();
  if (apiKey) authHeaders["X-API-Key"] = apiKey;
  if (options.body instanceof FormData) {
    request.headers = { ...authHeaders, ...(options.headers || {}) };
  } else {
    request.headers = { "Content-Type": "application/json", ...authHeaders, ...(options.headers || {}) };
  }
  const response = await fetch(path, request);
  if (!response.ok) {
    if (response.status === 401) openApiKeyModal("A valid API key is required.");
    const body = await response.text();
    try {
      const parsed = JSON.parse(body);
      throw new Error(parsed.detail || `Request failed: ${response.status}`);
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(body || `Request failed: ${response.status}`);
      throw error;
    }
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function showView(view) {
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  const label = { chat: "Knowledge Assistant", documents: "Document Explorer", evaluation: "Evaluation Center", system: "System Status" }[view];
  $("#page-label").textContent = label;
  if (view === "documents") renderDocuments();
  if (view === "evaluation") renderEvaluation();
  if (view === "system") renderSystem();
}

function addMessage(role, answer, payload = {}) {
  state.messages.push({ role, answer, payload });
  const container = $("#messages");
  if (state.messages.length === 1) container.innerHTML = "";
  const node = document.createElement("div");
  node.className = `message ${role}`;
  let html = `<div class="message-bubble">${escapeHtml(answer)}</div>`;
  if (role === "assistant" && payload.sources?.length) {
    html += `<div class="sources">${payload.sources.map((source) => `<div class="source-card"><strong>${escapeHtml(source.label)}</strong><span>${escapeHtml(source.title)}${source.page ? ` · page ${source.page}` : ""}</span><p>${escapeHtml(source.excerpt)}</p></div>`).join("")}</div>`;
  }
  if (role === "assistant" && payload.tools_used?.length) {
    html += `<div class="tool-pills">${payload.tools_used.map((tool) => `<span class="tool-pill">${escapeHtml(tool)}</span>`).join("")}</div>`;
  }
  node.innerHTML = html;
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
}

async function askQuestion(question) {
  if (!question.trim()) return;
  const input = $("#question-input");
  input.value = "";
  addMessage("user", question);
  const loading = document.createElement("div");
  loading.className = "message assistant";
  loading.innerHTML = "<div class=\"message-bubble\">Searching evidence and preparing a grounded answer…</div>";
  $("#messages").appendChild(loading);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  try {
    const payload = await api("/ask", { method: "POST", body: JSON.stringify({ question, session_id: state.sessionId }) });
    loading.remove();
    addMessage("assistant", payload.answer, payload);
  } catch (error) {
    loading.remove();
    addMessage("assistant", `The assistant is temporarily unavailable. ${error.message}`);
  }
}

function renderDocuments() {
  const query = $("#document-search").value.toLowerCase();
  const department = $("#department-filter").value;
  const docs = state.documents.filter((doc) => {
    const metadata = doc.metadata || {};
    const matchesDepartment = !department || metadata.department === department;
    const searchable = `${doc.title} ${doc.document_id} ${metadata.department || ""} ${metadata.category || ""}`.toLowerCase();
    return matchesDepartment && searchable.includes(query);
  });
  $("#document-total").textContent = state.documents.length;
  $("#hero-doc-count").textContent = state.documents.length;
  $("#document-grid").innerHTML = docs.length ? docs.map((doc) => {
    const metadata = doc.metadata || {};
    return `<article class="document-card"><div class="doc-icon">▤</div><h3>${escapeHtml(doc.title)}</h3><p>${escapeHtml(doc.document_id)} · ${escapeHtml(metadata.effective_date || "Internal source")}</p><div class="doc-tags"><span>${escapeHtml(metadata.department || "General")}</span><span>${escapeHtml(metadata.category || "Policy")}</span></div></article>`;
  }).join("") : "<div class=\"empty-state\">No matching documents found.</div>";
}

function renderEvaluation() {
  const metrics = state.metrics || {};
  const retrieval = metrics.retrieval || [];
  const cards = [
    { label: "Indexed documents", value: metrics.corpus?.documents ?? "—", note: "current knowledge layer" },
    { label: "Hybrid Recall@5", value: retrieval.find((row) => row.pipeline === "Hybrid")?.["recall@5"]?.toFixed(3) ?? "—", note: "measured retrieval" },
    { label: "RAG context support", value: metrics.rag?.context_support_rate?.toFixed(3) ?? "—", note: "offline evaluation" },
    { label: "Agent routing accuracy", value: metrics.agent?.routing_accuracy?.toFixed(3) ?? "—", note: "labeled routes" },
  ];
  $("#metric-cards").innerHTML = cards.map((card) => `<div class="metric-card"><span>${card.label}</span><strong>${card.value}</strong><small>${card.note}</small></div>`).join("");
  $("#retrieval-chart").innerHTML = retrieval.length ? retrieval.map((row) => `<div class="bar-row"><span>${escapeHtml(row.pipeline)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.round((row["recall@5"] || 0) * 100)}%"></div></div><b class="bar-value">${((row["recall@5"] || 0) * 100).toFixed(1)}%</b></div>`).join("") : "<div class=\"empty-state\">Run the benchmark scripts to populate this view.</div>";
  const agent = metrics.agent || {};
  $("#agent-quality").innerHTML = `<div class="quality-row"><span>Routing accuracy</span><strong>${agent.routing_accuracy?.toFixed(3) ?? "—"}</strong></div><div class="quality-row"><span>Average tool calls</span><strong>${agent.average_tool_calls?.toFixed(1) ?? "—"}</strong></div><div class="quality-row"><span>Average retries</span><strong>${agent.average_retries?.toFixed(1) ?? "—"}</strong></div><div class="quality-row"><span>RAG refusal rate</span><strong>${metrics.rag?.refusal_rate?.toFixed(3) ?? "—"}</strong></div>`;
}

function renderSystem() {
  const health = state.health || {};
  $("#service-api").textContent = health.status === "ok" ? "Operational" : "Unavailable";
  $("#service-index").textContent = (health.chunks || 0) > 0 ? "Operational" : "Empty";
  $("#detail-embedding").textContent = health.embedding_model || "—";
  $("#detail-documents").textContent = state.metrics?.corpus?.documents ?? "—";
  $("#detail-chunks").textContent = state.metrics?.corpus?.chunks ?? "—";
}

function renderAttachmentPreview(fileList) {
  const files = [...fileList];
  const preview = $("#attachment-preview");
  preview.innerHTML = files.map((file) => {
    if (file.type.startsWith("image/")) {
      return `<div class="attachment-chip image-chip"><img src="${URL.createObjectURL(file)}" alt="" /><span>${escapeHtml(file.name)}</span></div>`;
    }
    return `<div class="attachment-chip"><span class="attachment-icon">▤</span><span>${escapeHtml(file.name)}</span></div>`;
  }).join("");
}

async function uploadDocuments(fileList, inputSelector = "#upload-input", statusSelector = "#upload-status") {
  const files = [...fileList];
  if (!files.length) return;
  const status = $(statusSelector);
  status.className = "upload-status active";
  let successful = 0;
  let indexed = 0;
  let queued = 0;
  let images = 0;
  const uploadedNames = [];
  const errors = [];
  for (const file of files) {
    status.textContent = `Indexing ${file.name}…`;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const result = await api("/documents/upload", { method: "POST", body: formData });
      successful += 1;
      if (result.kind === "image") images += 1;
      if (result.indexed) indexed += 1;
      if (result.status === "queued") queued += 1;
      uploadedNames.push(file.name);
    } catch (error) {
      errors.push(`${file.name}: ${error.message}`);
      status.textContent = errors[errors.length - 1];
    }
  }
  try {
    await refresh();
  } catch (error) {
    status.textContent = `Upload terminé, mais le rafraîchissement a échoué : ${error.message}`;
  }
  const summary = images
    ? `${successful}/${files.length} file(s) received · ${images} image attachment(s) stored`
    : queued
      ? `${successful}/${files.length} received · ${queued} queued for indexing`
      : `${indexed}/${files.length} document(s) indexed`;
  status.textContent = errors.length ? `${summary} · Erreur : ${errors.join(" | ")}` : summary;
  $(inputSelector).value = "";
  if (statusSelector === "#attachment-status" && uploadedNames.length) {
    const suffix = images ? "Image stockée comme pièce jointe visuelle." : queued ? "Document placé dans la file d’indexation." : "Document indexé et disponible pour vos questions.";
    addMessage("assistant", `Upload terminé : ${uploadedNames.join(", ")}. ${suffix}`);
  }
}

function setupVoiceRecognition() {
  const button = $("#voice-button");
  const status = $("#voice-status");
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    button.disabled = true;
    button.title = "Voice input is supported by Chrome and Edge";
    status.textContent = "Voice input: use Chrome or Edge";
    return;
  }
  const recognition = new SpeechRecognition();
  const input = $("#question-input");
  let baseText = "";
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = navigator.language?.toLowerCase().startsWith("fr") ? "fr-FR" : "en-US";
  recognition.onstart = () => {
    state.listening = true;
    baseText = input.value.trim();
    button.classList.add("recording");
    button.textContent = "■ Stop";
    status.textContent = "Listening… speak your question";
  };
  recognition.onresult = (event) => {
    let transcript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) transcript += event.results[index][0].transcript;
    input.value = [baseText, transcript.trim()].filter(Boolean).join(" ");
  };
  recognition.onerror = (event) => { status.textContent = `Voice input: ${event.error}`; };
  recognition.onend = () => {
    state.listening = false;
    button.classList.remove("recording");
    button.textContent = "◉ Voice";
    if (!status.textContent.startsWith("Voice input:")) status.textContent = "Voice ready";
  };
  state.recognition = recognition;
  button.addEventListener("click", () => {
    if (state.listening) recognition.stop();
    else recognition.start();
  });
}

async function refresh() {
  try {
    state.health = await api("/health");
    state.metrics = await api("/metrics");
    state.documents = await api("/documents");
    $("#api-label").textContent = "API operational";
    $("#hero-doc-count").textContent = state.metrics.corpus?.documents ?? state.documents.length;
    const departments = [...new Set(state.documents.map((doc) => doc.metadata?.department).filter(Boolean))].sort();
    $("#department-filter").innerHTML = `<option value="">All departments</option>${departments.map((department) => `<option>${escapeHtml(department)}</option>`).join("")}`;
    renderDocuments();
    renderEvaluation();
    renderSystem();
    return true;
  } catch (error) {
    $("#api-label").textContent = "API unavailable";
    $("#api-label").style.color = "var(--danger)";
    return false;
  }
}

$("#ask-form").addEventListener("submit", (event) => { event.preventDefault(); askQuestion($("#question-input").value); });
$("#question-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#ask-form").requestSubmit(); }
});
$$('[data-prompt]').forEach((button) => button.addEventListener("click", () => { $("#question-input").value = button.dataset.prompt; askQuestion(button.dataset.prompt); }));
$("#upload-button").addEventListener("click", () => $("#upload-input").click());
$("#upload-input").addEventListener("change", (event) => uploadDocuments(event.target.files));
$("#attachment-button").addEventListener("click", () => $("#attachment-input").click());
$("#attachment-input").addEventListener("change", (event) => {
  renderAttachmentPreview(event.target.files);
  uploadDocuments(event.target.files, "#attachment-input", "#attachment-status");
});
$$('.nav-item').forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
$("#document-search").addEventListener("input", renderDocuments);
$("#department-filter").addEventListener("change", renderDocuments);
$("#refresh-button").addEventListener("click", refresh);
$("#system-refresh").addEventListener("click", refresh);
$("#api-key-button").addEventListener("click", () => openApiKeyModal());
$("#api-key-close").addEventListener("click", closeApiKeyModal);
$("#api-key-modal").addEventListener("click", (event) => {
  if (event.target.id === "api-key-modal") closeApiKeyModal();
});
$("#api-key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const key = $("#api-key-input").value.trim();
  if (!key) {
    $("#api-key-status").textContent = "Enter an API key to continue.";
    return;
  }
  setApiKey(key);
  $("#api-key-status").textContent = "Checking key…";
  if (await refresh()) {
    $("#api-key-status").textContent = "Connected.";
    closeApiKeyModal();
  } else {
    $("#api-key-status").textContent = "The key was rejected or the API is unavailable.";
  }
});
$("#new-chat").addEventListener("click", () => {
  state.sessionId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `session-${Date.now()}`;
  state.messages = [];
  $("#messages").innerHTML = "<div class=\"welcome-message\"><div class=\"welcome-icon\">✦</div><h2>What would you like to know?</h2><p>Ask about policies, procedures, expenses, security, or structured company data.</p></div>";
});
setupVoiceRecognition();
refresh();
