// ── State ─────────────────────────────────────────────────────
let currentPipelineId = null;
let chatHistory = {};
let isQuerying = false;
let cleanupIngestProgress = null;

// ── DOM refs ──────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const ollamaStatus = $('#ollama-status');
const pipelineList = $('#pipeline-list');

const viewWelcome = $('#view-welcome');
const viewPipeline = $('#view-pipeline');

const pipelineName = $('#pipeline-name');
const pipelineEmbModel = $('#pipeline-emb-model');
const pipelineChatModel = $('#pipeline-chat-model');
const pipelineDocCount = $('#pipeline-doc-count');
const pipelineChunkCount = $('#pipeline-chunk-count');
const docList = $('#doc-list');
const chatMessages = $('#chat-messages');
const chatInput = $('#chat-input');
const btnSend = $('#btn-send');
const btnAddDocs = $('#btn-add-docs');
const btnDeletePipeline = $('#btn-delete-pipeline');
const btnClearChat = $('#btn-clear-chat');
const ingestStatus = $('#ingest-status');

const modalCreate = $('#modal-create');
const inputPipelineName = $('#input-pipeline-name');
const selectPreset = $('#select-preset');
const presetHint = $('#preset-hint');
const selectEmbModel = $('#select-emb-model');
const selectChatModel = $('#select-chat-model');
const inputChunkSize = $('#input-chunk-size');
const inputChunkOverlap = $('#input-chunk-overlap');
const inputTopK = $('#input-top-k');
const inputTemperature = $('#input-temperature');
const inputMinScore = $('#input-min-score');
const inputContextWindow = $('#input-context-window');

// Settings panel refs
const settingsPreset = $('#settings-preset');
const settingsChunkSize = $('#settings-chunk-size');
const settingsChunkOverlap = $('#settings-chunk-overlap');
const settingsTopK = $('#settings-top-k');
const settingsTemperature = $('#settings-temperature');
const settingsMinScore = $('#settings-min-score');
const settingsContextWindow = $('#settings-context-window');
const settingsSystemPrompt = $('#settings-system-prompt');
const btnSaveSettings = $('#btn-save-settings');
const btnResetPreset = $('#btn-reset-preset');
const settingsSaveStatus = $('#settings-save-status');

const modalModels = $('#modal-models');
const modelList = $('#model-list');
const inputPullModel = $('#input-pull-model');
const pullStatus = $('#pull-status');

const toastContainer = $('#toast-container');
const dragOverlay = $('#drag-overlay');

// ── Toasts ────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

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

function renderMarkdown(text) {
  return text
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/^\- (.+)$/gm, '<li>$1</li>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/<\/ul>\s*<ul>/g, '')
    .replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>')
    .replace(/\n/g, '<br>');
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1e6) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1e9) return (bytes / 1e6).toFixed(1) + ' MB';
  return (bytes / 1e9).toFixed(1) + ' GB';
}

// ── Presets ───────────────────────────────────────────────────

let docPresets = {};

async function loadPresets() {
  docPresets = await window.api.listPresets();
}

function populatePresetSelect(selectEl, selectedValue) {
  selectEl.innerHTML = '';
  for (const [key, preset] of Object.entries(docPresets)) {
    const opt = new Option(preset.label, key);
    selectEl.add(opt);
  }
  if (selectedValue && docPresets[selectedValue]) {
    selectEl.value = selectedValue;
  }
}

function applyPresetToCreateModal(presetKey) {
  const preset = docPresets[presetKey];
  if (!preset) return;
  inputChunkSize.value = preset.chunkSize;
  inputChunkOverlap.value = preset.chunkOverlap;
  inputTopK.value = preset.topK;
  inputTemperature.value = preset.temperature;
  inputMinScore.value = preset.minScore;
  inputContextWindow.value = preset.contextWindow;
  presetHint.textContent = preset.description;
}

loadPresets();

// ── Ollama Status ─────────────────────────────────────────────

let ollamaOnline = false;

async function checkOllama() {
  const res = await window.api.ollamaStatus();
  const wasOffline = !ollamaOnline;
  ollamaOnline = res.running;
  if (res.running) {
    ollamaStatus.className = 'status-badge online';
    ollamaStatus.textContent = 'Ollama running';
    if (wasOffline) showToast('Ollama connected', 'success');
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

  if (pipelines.length === 0) {
    pipelineList.innerHTML =
      '<li class="pipeline-empty">No pipelines yet. Click + to create one.</li>';
    return;
  }

  for (const p of pipelines) {
    const li = document.createElement('li');
    li.className = 'pipeline-item' + (p.id === currentPipelineId ? ' active' : '');
    li.innerHTML = `
      <div class="pipeline-item-info">
        <span class="pipeline-item-name">${escapeHtml(p.name)}</span>
        <span class="pipeline-item-meta">${p.documents.length} docs &middot; ${p.chunkCount || 0} chunks</span>
      </div>
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
  pipelineDocCount.textContent = `${p.documents.length} docs`;
  pipelineChunkCount.textContent = `${p.chunkCount || 0} chunks`;

  // Populate settings panel
  populatePresetSelect(settingsPreset, p.preset || 'general');
  settingsChunkSize.value = p.chunkSize || 512;
  settingsChunkOverlap.value = p.chunkOverlap || 64;
  settingsTopK.value = p.topK || 5;
  settingsTemperature.value = p.temperature !== undefined ? p.temperature : 0.1;
  settingsMinScore.value = p.minScore !== undefined ? p.minScore : 0.3;
  settingsContextWindow.value = p.contextWindow || 6;
  settingsSystemPrompt.value = p.systemPrompt || '';
  settingsSaveStatus.textContent = '';

  await refreshDocuments();

  // Load persisted chat history
  const saved = await window.api.loadChatHistory(id);
  chatHistory[id] = saved || [];
  renderChat();

  showView('view-pipeline');
  refreshPipelines();
}

// ── Documents ─────────────────────────────────────────────────

async function refreshDocuments() {
  const docs = await window.api.listDocuments(currentPipelineId);
  docList.innerHTML = '';

  if (docs.length === 0) {
    docList.innerHTML =
      '<li class="doc-empty">No documents yet. Add some to get started.</li>';
    return;
  }

  for (const doc of docs) {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.innerHTML = `
      <svg class="doc-icon" viewBox="0 0 16 16" width="14" height="14">
        <path fill="currentColor" d="M3 1h7l4 4v9a1 1 0 01-1 1H3a1 1 0 01-1-1V2a1 1 0 011-1zm7 1.5V5h2.5L10 2.5zM3 2v12h10V6h-4V2H3z"/>
      </svg>
      <span class="doc-name">${escapeHtml(doc.fileName)}</span>
      <span class="doc-chunks">${doc.chunks} chunks</span>
      <button class="btn-icon btn-remove-doc" data-id="${doc.id}" title="Remove document">&times;</button>
    `;
    li.querySelector('.btn-remove-doc').addEventListener('click', async (e) => {
      e.stopPropagation();
      await window.api.removeDocument(currentPipelineId, doc.id);
      showToast(`Removed ${doc.fileName}`, 'info');
      await refreshDocuments();
      const p = await window.api.getPipeline(currentPipelineId);
      pipelineDocCount.textContent = `${p.documents.length} docs`;
      pipelineChunkCount.textContent = `${p.chunkCount || 0} chunks`;
      refreshPipelines();
    });
    docList.appendChild(li);
  }
}

// Ingestion progress
cleanupIngestProgress = window.api.onIngestProgress((progress) => {
  if (progress.type === 'file-start') {
    ingestStatus.textContent = `Processing ${progress.fileName} (${progress.fileIndex + 1}/${progress.totalFiles})...`;
    ingestStatus.classList.add('visible');
  } else if (progress.type === 'embedding-progress') {
    ingestStatus.textContent = `Embedding ${progress.fileName}: ${progress.chunksProcessed}/${progress.totalChunks} chunks`;
  } else if (progress.type === 'file-done') {
    if (progress.fileIndex === progress.totalFiles - 1) {
      ingestStatus.classList.remove('visible');
    }
  }
});

async function addDocuments() {
  if (!currentPipelineId) return;
  btnAddDocs.disabled = true;
  btnAddDocs.textContent = 'Ingesting...';
  try {
    const res = await window.api.addFiles(currentPipelineId);
    if (!res.canceled) {
      const total = res.reduce((s, r) => s + r.chunks, 0);
      showToast(`Added ${res.length} file(s), ${total} chunks`, 'success');
      await refreshDocuments();
      const p = await window.api.getPipeline(currentPipelineId);
      pipelineDocCount.textContent = `${p.documents.length} docs`;
      pipelineChunkCount.textContent = `${p.chunkCount || 0} chunks`;
      refreshPipelines();
    }
  } catch (err) {
    showToast(`Ingestion error: ${err.message}`, 'error');
  }
  btnAddDocs.disabled = false;
  btnAddDocs.textContent = 'Add Documents';
  ingestStatus.classList.remove('visible');
}

btnAddDocs.addEventListener('click', addDocuments);

// ── Drag and Drop ─────────────────────────────────────────────

let dragCounter = 0;

document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  if (!currentPipelineId) return;
  dragCounter++;
  dragOverlay.classList.add('visible');
});

document.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter <= 0) {
    dragCounter = 0;
    dragOverlay.classList.remove('visible');
  }
});

document.addEventListener('dragover', (e) => {
  e.preventDefault();
});

document.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragCounter = 0;
  dragOverlay.classList.remove('visible');
  if (!currentPipelineId) return;

  const paths = [];
  for (const file of e.dataTransfer.files) {
    if (file.path) paths.push(file.path);
  }

  if (paths.length === 0) return;

  btnAddDocs.disabled = true;
  btnAddDocs.textContent = 'Ingesting...';
  try {
    const res = await window.api.addFilePaths(currentPipelineId, paths);
    const total = res.reduce((s, r) => s + r.chunks, 0);
    showToast(`Added ${res.length} file(s), ${total} chunks`, 'success');
    await refreshDocuments();
    const p = await window.api.getPipeline(currentPipelineId);
    pipelineDocCount.textContent = `${p.documents.length} docs`;
    pipelineChunkCount.textContent = `${p.chunkCount || 0} chunks`;
    refreshPipelines();
  } catch (err) {
    showToast(`Ingestion error: ${err.message}`, 'error');
  }
  btnAddDocs.disabled = false;
  btnAddDocs.textContent = 'Add Documents';
  ingestStatus.classList.remove('visible');
});

// ── Chat ──────────────────────────────────────────────────────

function renderChat() {
  chatMessages.innerHTML = '';
  const history = chatHistory[currentPipelineId] || [];

  if (history.length === 0) {
    chatMessages.innerHTML = `
      <div class="chat-empty">
        <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
        </svg>
        <p>Ask a question about your documents</p>
      </div>
    `;
    return;
  }

  for (const msg of history) {
    addChatBubble(msg.role, msg.content, msg.sources);
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addChatBubble(role, content, sources) {
  // Remove empty state if present
  const empty = chatMessages.querySelector('.chat-empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;

  let html = `<div class="bubble-content">${renderMarkdown(content)}</div>`;
  if (sources && sources.length > 0) {
    html += '<div class="bubble-sources"><strong>Sources:</strong>';
    for (const s of sources) {
      html += `<div class="source-item">
        <span class="source-file">${escapeHtml(s.fileName)}</span>
        <span class="source-chunk">chunk ${s.chunkIndex !== undefined ? s.chunkIndex : '?'}</span>
        <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
      </div>`;
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
  // Persist
  window.api.saveChatHistory(currentPipelineId, chatHistory[currentPipelineId]);
}

async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || !currentPipelineId || isQuerying) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  addChatMessage('user', question);

  const bubble = addChatBubble('assistant', '');
  const contentEl = bubble.querySelector('.bubble-content');
  contentEl.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
  let fullText = '';

  isQuerying = true;
  btnSend.disabled = true;
  chatInput.disabled = true;
  btnSend.innerHTML = '<span class="spinner-small"></span>';

  // Build chat history for context (exclude sources)
  const historyForContext = (chatHistory[currentPipelineId] || [])
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .map((m) => ({ role: m.role, content: m.content }));

  try {
    await new Promise((resolve, reject) => {
      window.api.queryStream(
        currentPipelineId,
        question,
        historyForContext.slice(-6),
        (chunk) => {
          fullText += chunk;
          contentEl.innerHTML = renderMarkdown(fullText);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        (sources) => {
          // Add sources to the bubble
          if (sources && sources.length > 0) {
            let sourcesHtml = '<div class="bubble-sources"><strong>Sources:</strong>';
            for (const s of sources) {
              sourcesHtml += `<div class="source-item">
                <span class="source-file">${escapeHtml(s.fileName)}</span>
                <span class="source-chunk">chunk ${s.chunkIndex !== undefined ? s.chunkIndex : '?'}</span>
                <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
              </div>`;
            }
            sourcesHtml += '</div>';
            bubble.insertAdjacentHTML('beforeend', sourcesHtml);
          }

          // Save to history
          if (!chatHistory[currentPipelineId]) chatHistory[currentPipelineId] = [];
          chatHistory[currentPipelineId].push({
            role: 'assistant',
            content: fullText,
            sources,
          });
          window.api.saveChatHistory(currentPipelineId, chatHistory[currentPipelineId]);
          resolve();
        },
        (err) => reject(new Error(err))
      );
    });
  } catch (err) {
    contentEl.innerHTML = `<span class="error">Error: ${escapeHtml(err.message)}</span>`;
    showToast(`Query failed: ${err.message}`, 'error');
  }

  isQuerying = false;
  btnSend.disabled = false;
  chatInput.disabled = false;
  btnSend.textContent = 'Send';
  chatInput.focus();
}

btnSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
});

// Clear chat
btnClearChat.addEventListener('click', async () => {
  if (!currentPipelineId) return;
  chatHistory[currentPipelineId] = [];
  await window.api.clearChatHistory(currentPipelineId);
  renderChat();
  showToast('Chat cleared', 'info');
});

// ── Create Pipeline ───────────────────────────────────────────

async function openCreateModal() {
  const models = await window.api.ollamaModels();
  if (models.length === 0) {
    showToast('No models found. Pull models first via Manage Models.', 'error');
    return;
  }
  selectEmbModel.innerHTML = '';
  selectChatModel.innerHTML = '';
  for (const m of models) {
    const opt1 = new Option(m.name, m.name);
    const opt2 = new Option(m.name, m.name);
    selectEmbModel.add(opt1);
    selectChatModel.add(opt2);
  }
  const embDefault = models.find((m) => m.name.includes('embed'));
  if (embDefault) selectEmbModel.value = embDefault.name;
  const chatDefault = models.find(
    (m) => m.name.includes('llama') || m.name.includes('mistral') || m.name.includes('gemma')
  );
  if (chatDefault) selectChatModel.value = chatDefault.name;

  // Populate presets and apply default
  populatePresetSelect(selectPreset, 'general');
  applyPresetToCreateModal('general');

  inputPipelineName.value = '';
  modalCreate.classList.add('active');
  inputPipelineName.focus();
}

// Preset change in create modal
selectPreset.addEventListener('change', () => {
  applyPresetToCreateModal(selectPreset.value);
});

$('#btn-new-pipeline').addEventListener('click', openCreateModal);

$('#btn-cancel-create').addEventListener('click', () => {
  modalCreate.classList.remove('active');
});

$('#btn-confirm-create').addEventListener('click', async () => {
  const name = inputPipelineName.value.trim() || 'Untitled Pipeline';
  const embModel = selectEmbModel.value;
  const chatModel = selectChatModel.value;
  if (!embModel || !chatModel) return;

  const chunkSize = parseInt(inputChunkSize.value) || 512;
  const chunkOverlap = parseInt(inputChunkOverlap.value) || 64;
  const topK = parseInt(inputTopK.value) || 5;
  const temperature = parseFloat(inputTemperature.value);
  const minScore = parseFloat(inputMinScore.value);
  const contextWindow = parseInt(inputContextWindow.value) || 6;
  const preset = selectPreset.value;

  const p = await window.api.createPipeline(name, embModel, chatModel, {
    preset,
    chunkSize: Math.max(64, Math.min(2048, chunkSize)),
    chunkOverlap: Math.max(0, Math.min(512, chunkOverlap)),
    topK: Math.max(1, Math.min(20, topK)),
    temperature: Math.max(0, Math.min(2, isNaN(temperature) ? 0.1 : temperature)),
    minScore: Math.max(0, Math.min(1, isNaN(minScore) ? 0.3 : minScore)),
    contextWindow: Math.max(0, Math.min(20, contextWindow)),
  });
  modalCreate.classList.remove('active');
  showToast(`Pipeline "${name}" created`, 'success');
  await refreshPipelines();
  openPipeline(p.id);
});

// ── Pipeline Settings ──────────────────────────────────────────

btnSaveSettings.addEventListener('click', async () => {
  if (!currentPipelineId) return;
  const settings = {
    preset: settingsPreset.value,
    chunkSize: Math.max(64, Math.min(2048, parseInt(settingsChunkSize.value) || 512)),
    chunkOverlap: Math.max(0, Math.min(512, parseInt(settingsChunkOverlap.value) || 64)),
    topK: Math.max(1, Math.min(20, parseInt(settingsTopK.value) || 5)),
    temperature: Math.max(0, Math.min(2, parseFloat(settingsTemperature.value) || 0)),
    minScore: Math.max(0, Math.min(1, parseFloat(settingsMinScore.value) || 0)),
    contextWindow: Math.max(0, Math.min(20, parseInt(settingsContextWindow.value) || 6)),
    systemPrompt: settingsSystemPrompt.value,
  };
  await window.api.updatePipelineSettings(currentPipelineId, settings);
  settingsSaveStatus.textContent = 'Saved!';
  settingsSaveStatus.className = 'settings-save-status success';
  showToast('Settings saved. Changes apply to new queries.', 'success');
  setTimeout(() => { settingsSaveStatus.textContent = ''; }, 3000);
});

btnResetPreset.addEventListener('click', () => {
  const presetKey = settingsPreset.value;
  const preset = docPresets[presetKey];
  if (!preset) return;
  settingsChunkSize.value = preset.chunkSize;
  settingsChunkOverlap.value = preset.chunkOverlap;
  settingsTopK.value = preset.topK;
  settingsTemperature.value = preset.temperature;
  settingsMinScore.value = preset.minScore;
  settingsContextWindow.value = preset.contextWindow;
  settingsSystemPrompt.value = preset.systemPrompt;
  showToast(`Reset to "${preset.label}" defaults. Click Save to apply.`, 'info');
});

settingsPreset.addEventListener('change', () => {
  const presetKey = settingsPreset.value;
  const preset = docPresets[presetKey];
  if (!preset) return;
  settingsChunkSize.value = preset.chunkSize;
  settingsChunkOverlap.value = preset.chunkOverlap;
  settingsTopK.value = preset.topK;
  settingsTemperature.value = preset.temperature;
  settingsMinScore.value = preset.minScore;
  settingsContextWindow.value = preset.contextWindow;
  settingsSystemPrompt.value = preset.systemPrompt;
});

// ── Delete Pipeline ───────────────────────────────────────────

btnDeletePipeline.addEventListener('click', async () => {
  if (!currentPipelineId) return;
  if (!confirm('Delete this pipeline and all its data?')) return;
  const p = await window.api.getPipeline(currentPipelineId);
  await window.api.deletePipeline(currentPipelineId);
  delete chatHistory[currentPipelineId];
  currentPipelineId = null;
  showView('view-welcome');
  refreshPipelines();
  showToast(`Pipeline "${p?.name}" deleted`, 'info');
});

// ── Pipeline Rename (double-click) ────────────────────────────

pipelineName.addEventListener('dblclick', () => {
  const current = pipelineName.textContent;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = current;
  input.className = 'rename-input';
  pipelineName.replaceWith(input);
  input.focus();
  input.select();

  const commit = async () => {
    const newName = input.value.trim() || current;
    const h1 = document.createElement('h1');
    h1.id = 'pipeline-name';
    h1.textContent = newName;
    h1.title = 'Double-click to rename';
    h1.addEventListener('dblclick', arguments.callee);
    input.replaceWith(h1);
    // Re-set the reference
    Object.defineProperty(window, '_pipelineName', { value: h1 });
    if (newName !== current && currentPipelineId) {
      await window.api.renamePipeline(currentPipelineId, newName);
      refreshPipelines();
    }
  };

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      input.blur();
    }
    if (e.key === 'Escape') {
      input.value = current;
      input.blur();
    }
  });
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
    li.innerHTML = `
      <div class="model-info">
        <span class="model-name">${escapeHtml(m.name)}</span>
        <span class="model-details">${formatFileSize(m.size)}${m.details?.family ? ' &middot; ' + escapeHtml(m.details.family) : ''}</span>
      </div>
      <button class="btn-icon btn-delete-model" title="Delete model">&times;</button>
    `;
    li.querySelector('.btn-delete-model').addEventListener('click', async () => {
      if (!confirm(`Delete model "${m.name}"?`)) return;
      try {
        await window.api.ollamaDeleteModel(m.name);
        showToast(`Deleted ${m.name}`, 'info');
        await refreshModelList();
      } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
      }
    });
    modelList.appendChild(li);
  }
  if (models.length === 0) {
    modelList.innerHTML = '<li class="empty">No models installed. Pull one above.</li>';
  }
}

$('#btn-pull-model').addEventListener('click', pullModel);
inputPullModel.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') pullModel();
});

async function pullModel() {
  const name = inputPullModel.value.trim();
  if (!name) return;
  pullStatus.textContent = `Pulling ${name}... (this may take a while)`;
  pullStatus.className = 'pull-status loading';
  $('#btn-pull-model').disabled = true;
  try {
    await window.api.ollamaPull(name);
    pullStatus.textContent = `Successfully pulled ${name}`;
    pullStatus.className = 'pull-status success';
    inputPullModel.value = '';
    showToast(`Model "${name}" pulled successfully`, 'success');
    await refreshModelList();
  } catch (err) {
    pullStatus.textContent = `Error: ${err.message}`;
    pullStatus.className = 'pull-status error';
    showToast(`Failed to pull ${name}`, 'error');
  }
  $('#btn-pull-model').disabled = false;
}

// ── Menu shortcuts ────────────────────────────────────────────
window.api.onMenuNewPipeline(openCreateModal);
window.api.onMenuAddDocuments(addDocuments);

// ── Keyboard shortcuts ────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Escape closes modals
  if (e.key === 'Escape') {
    modalCreate.classList.remove('active');
    modalModels.classList.remove('active');
  }
});

// Close modals by clicking backdrop
modalCreate.addEventListener('click', (e) => {
  if (e.target === modalCreate) modalCreate.classList.remove('active');
});
modalModels.addEventListener('click', (e) => {
  if (e.target === modalModels) modalModels.classList.remove('active');
});

// ── Init ──────────────────────────────────────────────────────
refreshPipelines();
