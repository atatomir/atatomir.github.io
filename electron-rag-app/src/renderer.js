// ── State ─────────────────────────────────────────────────────
let currentPipelineId = null;
let chatHistory = {}; // pipelineId -> [{ role, content }]

// ── DOM refs ──────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const ollamaStatus = $('#ollama-status');
const pipelineList = $('#pipeline-list');
const mainContent = $('#main-content');

const viewWelcome = $('#view-welcome');
const viewPipeline = $('#view-pipeline');

const pipelineName = $('#pipeline-name');
const pipelineEmbModel = $('#pipeline-emb-model');
const pipelineChatModel = $('#pipeline-chat-model');
const pipelineDocCount = $('#pipeline-doc-count');
const docList = $('#doc-list');
const chatMessages = $('#chat-messages');
const chatInput = $('#chat-input');
const btnSend = $('#btn-send');
const btnAddDocs = $('#btn-add-docs');
const btnDeletePipeline = $('#btn-delete-pipeline');

const modalCreate = $('#modal-create');
const inputPipelineName = $('#input-pipeline-name');
const selectEmbModel = $('#select-emb-model');
const selectChatModel = $('#select-chat-model');

const modalModels = $('#modal-models');
const modelList = $('#model-list');
const inputPullModel = $('#input-pull-model');
const pullStatus = $('#pull-status');

// ── Helpers ───────────────────────────────────────────────────

function showView(id) {
  document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderMarkdownLite(text) {
  // Very basic: code blocks, bold, line breaks
  return text
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

// ── Ollama Status ─────────────────────────────────────────────

async function checkOllama() {
  const res = await window.api.ollamaStatus();
  if (res.running) {
    ollamaStatus.className = 'status-badge online';
    ollamaStatus.textContent = 'Ollama running';
  } else {
    ollamaStatus.className = 'status-badge offline';
    ollamaStatus.textContent = 'Ollama offline';
  }
}

setInterval(checkOllama, 5000);
checkOllama();

// ── Pipeline List ─────────────────────────────────────────────

async function refreshPipelines() {
  const pipelines = await window.api.listPipelines();
  pipelineList.innerHTML = '';
  for (const p of pipelines) {
    const li = document.createElement('li');
    li.className = 'pipeline-item' + (p.id === currentPipelineId ? ' active' : '');
    li.innerHTML = `
      <span class="pipeline-item-name">${escapeHtml(p.name)}</span>
      <span class="pipeline-item-docs">${p.documents.length} docs</span>
    `;
    li.addEventListener('click', () => openPipeline(p.id));
    pipelineList.appendChild(li);
  }
}

// ── Open Pipeline ─────────────────────────────────────────────

async function openPipeline(id) {
  currentPipelineId = id;
  const p = await window.api.getPipeline(id);
  if (!p) return;

  pipelineName.textContent = p.name;
  pipelineEmbModel.textContent = `Embed: ${p.embeddingModel}`;
  pipelineChatModel.textContent = `Chat: ${p.chatModel}`;
  pipelineDocCount.textContent = `${p.documents.length} documents`;

  await refreshDocuments();
  renderChat();
  showView('view-pipeline');
  refreshPipelines();
}

// ── Documents ─────────────────────────────────────────────────

async function refreshDocuments() {
  const docs = await window.api.listDocuments(currentPipelineId);
  docList.innerHTML = '';
  for (const doc of docs) {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.innerHTML = `
      <span class="doc-name">${escapeHtml(doc.fileName)}</span>
      <span class="doc-chunks">${doc.chunks} chunks</span>
      <button class="btn-icon btn-remove-doc" data-id="${doc.id}" title="Remove">&times;</button>
    `;
    li.querySelector('.btn-remove-doc').addEventListener('click', async () => {
      await window.api.removeDocument(currentPipelineId, doc.id);
      await refreshDocuments();
      const p = await window.api.getPipeline(currentPipelineId);
      pipelineDocCount.textContent = `${p.documents.length} documents`;
      refreshPipelines();
    });
    docList.appendChild(li);
  }
}

btnAddDocs.addEventListener('click', async () => {
  btnAddDocs.disabled = true;
  btnAddDocs.textContent = 'Ingesting...';
  try {
    const res = await window.api.addFiles(currentPipelineId);
    if (!res.canceled) {
      await refreshDocuments();
      const p = await window.api.getPipeline(currentPipelineId);
      pipelineDocCount.textContent = `${p.documents.length} documents`;
      refreshPipelines();
    }
  } catch (err) {
    addChatMessage('system', `Error ingesting files: ${err.message}`);
  }
  btnAddDocs.disabled = false;
  btnAddDocs.textContent = 'Add Documents';
});

// ── Chat ──────────────────────────────────────────────────────

function renderChat() {
  chatMessages.innerHTML = '';
  const history = chatHistory[currentPipelineId] || [];
  for (const msg of history) {
    addChatBubble(msg.role, msg.content, msg.sources);
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addChatBubble(role, content, sources) {
  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;

  let html = `<div class="bubble-content">${renderMarkdownLite(content)}</div>`;
  if (sources && sources.length > 0) {
    html += '<div class="bubble-sources"><strong>Sources:</strong>';
    for (const s of sources) {
      html += `<div class="source-item"><span class="source-file">${escapeHtml(s.fileName)}</span> <span class="source-score">(${(s.score * 100).toFixed(1)}%)</span></div>`;
    }
    html += '</div>';
  }
  div.innerHTML = html;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function addChatMessage(role, content, sources) {
  if (!chatHistory[currentPipelineId]) chatHistory[currentPipelineId] = [];
  chatHistory[currentPipelineId].push({ role, content, sources });
  addChatBubble(role, content, sources);
}

async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || !currentPipelineId) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  addChatMessage('user', question);

  // Create a streaming assistant bubble
  const bubble = addChatBubble('assistant', '');
  const contentEl = bubble.querySelector('.bubble-content');
  let fullText = '';

  btnSend.disabled = true;
  chatInput.disabled = true;

  try {
    await new Promise((resolve, reject) => {
      window.api.queryStream(
        currentPipelineId,
        question,
        (chunk) => {
          fullText += chunk;
          contentEl.innerHTML = renderMarkdownLite(fullText);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        () => resolve(),
        (err) => reject(new Error(err))
      );
    });

    // Save to history
    if (!chatHistory[currentPipelineId]) chatHistory[currentPipelineId] = [];
    chatHistory[currentPipelineId].push({ role: 'assistant', content: fullText });
  } catch (err) {
    contentEl.innerHTML = `<span class="error">Error: ${escapeHtml(err.message)}</span>`;
  }

  btnSend.disabled = false;
  chatInput.disabled = false;
  chatInput.focus();
}

btnSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// ── Create Pipeline ───────────────────────────────────────────

$('#btn-new-pipeline').addEventListener('click', async () => {
  const models = await window.api.ollamaModels();
  selectEmbModel.innerHTML = '';
  selectChatModel.innerHTML = '';
  for (const m of models) {
    selectEmbModel.innerHTML += `<option value="${m.name}">${m.name}</option>`;
    selectChatModel.innerHTML += `<option value="${m.name}">${m.name}</option>`;
  }
  // Pre-select reasonable defaults
  const embDefault = models.find((m) => m.name.includes('embed'));
  if (embDefault) selectEmbModel.value = embDefault.name;
  const chatDefault = models.find((m) => m.name.includes('llama') || m.name.includes('mistral'));
  if (chatDefault) selectChatModel.value = chatDefault.name;

  inputPipelineName.value = '';
  modalCreate.classList.add('active');
});

$('#btn-cancel-create').addEventListener('click', () => {
  modalCreate.classList.remove('active');
});

$('#btn-confirm-create').addEventListener('click', async () => {
  const name = inputPipelineName.value.trim() || 'Untitled Pipeline';
  const embModel = selectEmbModel.value;
  const chatModel = selectChatModel.value;
  if (!embModel || !chatModel) return;

  const p = await window.api.createPipeline(name, embModel, chatModel);
  modalCreate.classList.remove('active');
  await refreshPipelines();
  openPipeline(p.id);
});

// ── Delete Pipeline ───────────────────────────────────────────

btnDeletePipeline.addEventListener('click', async () => {
  if (!currentPipelineId) return;
  if (!confirm('Delete this pipeline and all its data?')) return;
  await window.api.deletePipeline(currentPipelineId);
  delete chatHistory[currentPipelineId];
  currentPipelineId = null;
  showView('view-welcome');
  refreshPipelines();
});

// ── Models Modal ──────────────────────────────────────────────

$('#btn-manage-models').addEventListener('click', async () => {
  await refreshModelList();
  modalModels.classList.add('active');
});

$('#btn-close-models').addEventListener('click', () => {
  modalModels.classList.remove('active');
});

async function refreshModelList() {
  const models = await window.api.ollamaModels();
  modelList.innerHTML = '';
  for (const m of models) {
    const li = document.createElement('li');
    const sizeMB = (m.size / 1e6).toFixed(0);
    li.innerHTML = `<span class="model-name">${escapeHtml(m.name)}</span><span class="model-size">${sizeMB} MB</span>`;
    modelList.appendChild(li);
  }
  if (models.length === 0) {
    modelList.innerHTML = '<li class="empty">No models installed. Pull one above.</li>';
  }
}

$('#btn-pull-model').addEventListener('click', async () => {
  const name = inputPullModel.value.trim();
  if (!name) return;
  pullStatus.textContent = `Pulling ${name}...`;
  pullStatus.className = 'pull-status loading';
  try {
    await window.api.ollamaPull(name);
    pullStatus.textContent = `Successfully pulled ${name}`;
    pullStatus.className = 'pull-status success';
    inputPullModel.value = '';
    await refreshModelList();
  } catch (err) {
    pullStatus.textContent = `Error: ${err.message}`;
    pullStatus.className = 'pull-status error';
  }
});

// ── Init ──────────────────────────────────────────────────────
refreshPipelines();
