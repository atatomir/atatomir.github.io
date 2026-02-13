const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { RagEngine } = require('./rag-engine');

let mainWindow;
let ragEngine;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0f0f13',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(async () => {
  const userDataPath = app.getPath('userData');
  ragEngine = new RagEngine(userDataPath);
  await ragEngine.init();

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ── IPC Handlers ──────────────────────────────────────────────

// Ollama status
ipcMain.handle('ollama:status', async () => {
  return ragEngine.checkOllamaStatus();
});

ipcMain.handle('ollama:models', async () => {
  return ragEngine.listModels();
});

ipcMain.handle('ollama:pull', async (_e, modelName) => {
  return ragEngine.pullModel(modelName);
});

// Pipeline CRUD
ipcMain.handle('pipeline:list', async () => {
  return ragEngine.listPipelines();
});

ipcMain.handle('pipeline:create', async (_e, name, embeddingModel, chatModel) => {
  return ragEngine.createPipeline(name, embeddingModel, chatModel);
});

ipcMain.handle('pipeline:delete', async (_e, id) => {
  return ragEngine.deletePipeline(id);
});

ipcMain.handle('pipeline:get', async (_e, id) => {
  return ragEngine.getPipeline(id);
});

// Document management
ipcMain.handle('pipeline:add-files', async (_e, pipelineId) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Documents', extensions: ['pdf', 'txt', 'md', 'docx', 'json', 'csv'] },
    ],
  });
  if (result.canceled) return { canceled: true };
  return ragEngine.ingestFiles(pipelineId, result.filePaths);
});

ipcMain.handle('pipeline:documents', async (_e, pipelineId) => {
  return ragEngine.listDocuments(pipelineId);
});

ipcMain.handle('pipeline:remove-document', async (_e, pipelineId, docId) => {
  return ragEngine.removeDocument(pipelineId, docId);
});

// Query
ipcMain.handle('pipeline:query', async (_e, pipelineId, question) => {
  return ragEngine.query(pipelineId, question);
});

// Streaming query
ipcMain.on('pipeline:query-stream', async (event, pipelineId, question) => {
  try {
    await ragEngine.queryStream(pipelineId, question, (chunk) => {
      event.reply('pipeline:query-stream-chunk', chunk);
    });
    event.reply('pipeline:query-stream-done');
  } catch (err) {
    event.reply('pipeline:query-stream-error', err.message);
  }
});
