const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Ollama
  ollamaStatus: () => ipcRenderer.invoke('ollama:status'),
  ollamaModels: () => ipcRenderer.invoke('ollama:models'),
  ollamaPull: (model) => ipcRenderer.invoke('ollama:pull', model),

  // Pipelines
  listPipelines: () => ipcRenderer.invoke('pipeline:list'),
  createPipeline: (name, embModel, chatModel) =>
    ipcRenderer.invoke('pipeline:create', name, embModel, chatModel),
  deletePipeline: (id) => ipcRenderer.invoke('pipeline:delete', id),
  getPipeline: (id) => ipcRenderer.invoke('pipeline:get', id),

  // Documents
  addFiles: (pipelineId) => ipcRenderer.invoke('pipeline:add-files', pipelineId),
  listDocuments: (pipelineId) => ipcRenderer.invoke('pipeline:documents', pipelineId),
  removeDocument: (pipelineId, docId) =>
    ipcRenderer.invoke('pipeline:remove-document', pipelineId, docId),

  // Query
  query: (pipelineId, question) => ipcRenderer.invoke('pipeline:query', pipelineId, question),

  // Streaming query
  queryStream: (pipelineId, question, onChunk, onDone, onError) => {
    const chunkHandler = (_e, chunk) => onChunk(chunk);
    const doneHandler = () => {
      ipcRenderer.removeListener('pipeline:query-stream-chunk', chunkHandler);
      ipcRenderer.removeListener('pipeline:query-stream-done', doneHandler);
      ipcRenderer.removeListener('pipeline:query-stream-error', errorHandler);
      onDone();
    };
    const errorHandler = (_e, msg) => {
      ipcRenderer.removeListener('pipeline:query-stream-chunk', chunkHandler);
      ipcRenderer.removeListener('pipeline:query-stream-done', doneHandler);
      ipcRenderer.removeListener('pipeline:query-stream-error', errorHandler);
      onError(msg);
    };
    ipcRenderer.on('pipeline:query-stream-chunk', chunkHandler);
    ipcRenderer.on('pipeline:query-stream-done', doneHandler);
    ipcRenderer.on('pipeline:query-stream-error', errorHandler);
    ipcRenderer.send('pipeline:query-stream', pipelineId, question);
  },
});
