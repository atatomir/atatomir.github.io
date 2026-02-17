const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Presets
  listPresets: () => ipcRenderer.invoke('presets:list'),

  // Ollama
  ollamaStatus: () => ipcRenderer.invoke('ollama:status'),
  ollamaModels: () => ipcRenderer.invoke('ollama:models'),
  ollamaPull: (model) => ipcRenderer.invoke('ollama:pull', model),
  ollamaDeleteModel: (model) => ipcRenderer.invoke('ollama:delete-model', model),

  // Pipelines
  listPipelines: () => ipcRenderer.invoke('pipeline:list'),
  createPipeline: (name, embModel, chatModel, opts) =>
    ipcRenderer.invoke('pipeline:create', name, embModel, chatModel, opts),
  deletePipeline: (id) => ipcRenderer.invoke('pipeline:delete', id),
  getPipeline: (id) => ipcRenderer.invoke('pipeline:get', id),
  renamePipeline: (id, name) => ipcRenderer.invoke('pipeline:rename', id, name),
  updatePipelineSettings: (id, settings) =>
    ipcRenderer.invoke('pipeline:update-settings', id, settings),

  // Documents
  addFiles: (pipelineId) => ipcRenderer.invoke('pipeline:add-files', pipelineId),
  addFilePaths: (pipelineId, paths) =>
    ipcRenderer.invoke('pipeline:add-file-paths', pipelineId, paths),
  listDocuments: (pipelineId) => ipcRenderer.invoke('pipeline:documents', pipelineId),
  removeDocument: (pipelineId, docId) =>
    ipcRenderer.invoke('pipeline:remove-document', pipelineId, docId),

  // Ingestion progress
  onIngestProgress: (callback) => {
    const handler = (_e, progress) => callback(progress);
    ipcRenderer.on('pipeline:ingest-progress', handler);
    return () => ipcRenderer.removeListener('pipeline:ingest-progress', handler);
  },

  // Chat history persistence
  loadChatHistory: (pipelineId) => ipcRenderer.invoke('chat:load', pipelineId),
  saveChatHistory: (pipelineId, messages) =>
    ipcRenderer.invoke('chat:save', pipelineId, messages),
  clearChatHistory: (pipelineId) => ipcRenderer.invoke('chat:clear', pipelineId),

  // Streaming query
  queryStream: (pipelineId, question, chatMessages, onChunk, onDone, onError) => {
    const chunkHandler = (_e, chunk) => onChunk(chunk);
    const doneHandler = (_e, sources) => {
      cleanup();
      onDone(sources);
    };
    const errorHandler = (_e, msg) => {
      cleanup();
      onError(msg);
    };
    const cleanup = () => {
      ipcRenderer.removeListener('pipeline:query-stream-chunk', chunkHandler);
      ipcRenderer.removeListener('pipeline:query-stream-done', doneHandler);
      ipcRenderer.removeListener('pipeline:query-stream-error', errorHandler);
    };
    ipcRenderer.on('pipeline:query-stream-chunk', chunkHandler);
    ipcRenderer.on('pipeline:query-stream-done', doneHandler);
    ipcRenderer.on('pipeline:query-stream-error', errorHandler);
    ipcRenderer.send('pipeline:query-stream', pipelineId, question, chatMessages);
    return cleanup;
  },

  // Menu events
  onMenuNewPipeline: (callback) => {
    ipcRenderer.on('menu:new-pipeline', callback);
    return () => ipcRenderer.removeListener('menu:new-pipeline', callback);
  },
  onMenuAddDocuments: (callback) => {
    ipcRenderer.on('menu:add-documents', callback);
    return () => ipcRenderer.removeListener('menu:add-documents', callback);
  },
});
