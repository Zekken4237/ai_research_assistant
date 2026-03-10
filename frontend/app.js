const STORAGE_KEY = "ai-research-assistant-ui";
const API_BASE_CANDIDATES = window.location.protocol.startsWith("http")
  ? [window.location.origin, "http://127.0.0.1:8000"]
  : ["http://127.0.0.1:8000"];

const state = {
  apiBase: API_BASE_CANDIDATES[0],
  apiReachable: true,
  modelAvailable: true,
  modelName: "llama2-uncensored:7b",
  paperName: "No paper loaded",
  ready: false,
  asking: false,
  uploading: false,
  history: [],
};

const elements = {
  apiStatus: document.getElementById("api-status"),
  askButton: document.getElementById("ask-button"),
  composerHint: document.getElementById("composer-hint"),
  conversation: document.getElementById("conversation"),
  jumpToWorkbench: document.getElementById("jump-to-workbench"),
  paperName: document.getElementById("paper-name"),
  paperStatus: document.getElementById("paper-status"),
  pdfFile: document.getElementById("pdf-file"),
  questionForm: document.getElementById("question-form"),
  questionInput: document.getElementById("question-input"),
  selectedFileName: document.getElementById("selected-file-name"),
  threadCaption: document.getElementById("thread-caption"),
  uploadButton: document.getElementById("upload-button"),
  uploadForm: document.getElementById("upload-form"),
};

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.paperName = saved.paperName || state.paperName;
    state.ready = Boolean(saved.ready);
    state.history = Array.isArray(saved.history) ? saved.history : [];
  } catch {
    state.history = [];
  }

  if (state.history.length === 0) {
    state.history.push({
      role: "assistant",
      content:
        "Upload a paper to begin. Once it is indexed, ask for summaries, findings, methodology, limitations, or presentation-ready explanations.",
      timestamp: new Date().toISOString(),
    });
  }
}

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      paperName: state.paperName,
      ready: state.ready,
      history: state.history.slice(-20),
    }),
  );
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function setApiStatus(mode, text) {
  elements.apiStatus.className = `status-pill status-pill--${mode}`;
  elements.apiStatus.textContent = text;
}

function updateControls() {
  const askDisabled = !state.ready || state.asking || !state.apiReachable || !state.modelAvailable;
  const uploadDisabled = state.uploading || !state.apiReachable;

  elements.askButton.disabled = askDisabled;
  elements.questionInput.disabled = askDisabled;
  elements.uploadButton.disabled = uploadDisabled;
  elements.pdfFile.disabled = uploadDisabled;

  if (state.uploading) {
    elements.paperStatus.textContent = "Processing paper";
    elements.uploadButton.textContent = "Processing...";
  } else {
    elements.uploadButton.textContent = "Process paper";
    elements.paperStatus.textContent = state.ready ? "Ready for questions" : "Waiting for upload";
  }

  if (state.asking) {
    elements.askButton.textContent = "Thinking...";
    elements.composerHint.textContent = `Reading the indexed paper with ${state.modelName} and drafting an answer.`;
  } else if (!state.apiReachable) {
    elements.askButton.textContent = "Ask paper";
    elements.composerHint.textContent = "Backend unavailable. Start the API server and refresh the page.";
  } else if (!state.modelAvailable) {
    elements.askButton.textContent = "Ask paper";
    elements.composerHint.textContent = `Start Ollama and make sure the ${state.modelName} model is available.`;
  } else {
    elements.askButton.textContent = "Ask paper";
    elements.composerHint.textContent = state.ready
      ? `Ask direct, specific questions for better answers from ${state.modelName}.`
      : "Upload a PDF before sending a question.";
  }

  elements.paperName.textContent = state.paperName;
  elements.threadCaption.textContent = state.ready
    ? `Currently indexed: ${state.paperName}`
    : "Upload a paper to unlock the question composer.";
}

function renderConversation() {
  elements.conversation.innerHTML = "";

  state.history.forEach((entry) => {
    const article = document.createElement("article");
    article.className = `message message--${entry.role}`;

    const meta = document.createElement("div");
    meta.className = "message__meta";

    const label = document.createElement("span");
    label.className = "message__label";
    label.textContent =
      entry.role === "user"
        ? "You"
        : entry.role === "system"
          ? "Status"
          : "Assistant";

    const time = document.createElement("span");
    time.textContent = formatTime(entry.timestamp);

    const content = document.createElement("div");
    content.className = "message__content";
    content.textContent = entry.content;

    meta.append(label, time);
    article.append(meta, content);
    elements.conversation.append(article);
  });

  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function addMessage(role, content) {
  state.history.push({
    role,
    content,
    timestamp: new Date().toISOString(),
  });
  state.history = state.history.slice(-20);
  persistState();
  renderConversation();
}

async function fetchJson(path, options = {}) {
  let lastError;
  const candidates = [...new Set([state.apiBase, ...API_BASE_CANDIDATES])];

  for (const base of candidates) {
    try {
      const response = await fetch(`${base}${path}`, options);
      if (response.status === 404 && candidates.length > 1 && base !== candidates[candidates.length - 1]) {
        continue;
      }

      const data = await response.json().catch(() => ({}));
      state.apiBase = base;
      return { response, data };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("The API is unavailable.");
}

async function checkHealth() {
  setApiStatus("checking", "Checking API");

  try {
    const { response, data } = await fetchJson("/health");
    if (!response.ok) {
      throw new Error("API returned an unexpected response.");
    }
    state.apiReachable = true;
    state.modelAvailable = data.ollama_available !== false;
    state.modelName = data.ollama_model || state.modelName;
    setApiStatus(
      state.modelAvailable ? "online" : "warning",
      state.modelAvailable ? `${state.modelName} ready` : `${state.modelName} unavailable`,
    );
    updateControls();
  } catch {
    state.apiReachable = false;
    setApiStatus("offline", "API unavailable");
    updateControls();
  }
}

async function uploadPaper(event) {
  event.preventDefault();

  const [file] = elements.pdfFile.files;
  if (!file) {
    addMessage("system", "Choose a PDF file before trying to process it.");
    return;
  }

  state.uploading = true;
  updateControls();
  const previousPaperName = state.paperName;
  const previousReady = state.ready;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const { response, data } = await fetchJson("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(data.detail || "The paper could not be processed.");
    }

    state.paperName = data.filename || file.name;
    state.ready = true;
    persistState();
    addMessage("system", `Indexed "${state.paperName}". You can start asking questions now.`);
    elements.questionInput.focus();
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Upload failed.";
    state.paperName = previousPaperName;
    state.ready = previousReady;
    addMessage("system", detail);
  } finally {
    state.uploading = false;
    updateControls();
    checkHealth();
  }
}

async function askQuestion(event) {
  event.preventDefault();

  const question = elements.questionInput.value.trim();
  if (!question || !state.ready) {
    return;
  }

  state.asking = true;
  updateControls();
  addMessage("user", question);
  elements.questionInput.value = "";

  try {
    const query = new URLSearchParams({ question });
    const { response, data } = await fetchJson(`/ask?${query}`, { method: "POST" });

    if (!response.ok) {
      throw new Error(data.detail || "The assistant could not answer the question.");
    }

    addMessage("assistant", data.answer);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Question failed.";
    elements.questionInput.value = question;
    addMessage("system", detail);
  } finally {
    state.asking = false;
    updateControls();
  }
}

function applyPrompt(prompt) {
  elements.questionInput.value = prompt;
  elements.questionInput.focus();
}

function bindEvents() {
  elements.jumpToWorkbench.addEventListener("click", () => {
    document.getElementById("workbench").scrollIntoView({ behavior: "smooth" });
  });

  elements.pdfFile.addEventListener("change", () => {
    const [file] = elements.pdfFile.files;
    elements.selectedFileName.textContent = file
      ? file.name
      : "Choose a paper or drag one here";
  });

  elements.uploadForm.addEventListener("submit", uploadPaper);
  elements.questionForm.addEventListener("submit", askQuestion);

  elements.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.questionForm.requestSubmit();
    }
  });

  document.querySelectorAll(".prompt-chip").forEach((button) => {
    button.addEventListener("click", () => {
      applyPrompt(button.dataset.prompt || "");
    });
  });
}

function init() {
  loadState();
  updateControls();
  renderConversation();
  bindEvents();
  checkHealth();
}

init();
