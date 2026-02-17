const { app, BrowserWindow, ipcMain, dialog, Menu, shell } = require('electron');
const path = require('path');
const { RagEngine, DOC_PRESETS } = require('./rag-engine');

let mainWindow;
let ragEngine;
let activeQueryAbort = null;

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
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

function buildMenu() {
  const template = [
    {
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'File',
      submenu: [
        {
          label: 'New Pipeline',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow?.webContents.send('menu:new-pipeline'),
        },
        {
          label: 'Add Documents',
          accelerator: 'CmdOrCtrl+O',
          click: () => mainWindow?.webContents.send('menu:add-documents'),
        },
        { type: 'separator' },
        { role: 'close' },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'front' },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Ollama Website',
          click: () => shell.openExternal('https://ollama.com'),
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// Wrap IPC handlers with error handling
function handle(channel, handler) {
  ipcMain.handle(channel, async (event, ...args) => {
    try {
      return await handler(event, ...args);
    } catch (err) {
      throw new Error(err.message || String(err));
    }
  });
}

app.whenReady().then(async () => {
  const userDataPath = app.getPath('userData');
  ragEngine = new RagEngine(userDataPath);
  await ragEngine.init();

  buildMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ── IPC Handlers ──────────────────────────────────────────────

// Presets
handle('presets:list', () => DOC_PRESETS);

// Ollama
handle('ollama:status', () => ragEngine.checkOllamaStatus());
handle('ollama:models', () => ragEngine.listModels());
handle('ollama:pull', (_e, modelName) => ragEngine.pullModel(modelName));
handle('ollama:delete-model', (_e, modelName) => ragEngine.deleteModel(modelName));

// Pipeline CRUD
handle('pipeline:list', () => ragEngine.listPipelines());
handle('pipeline:create', (_e, name, embeddingModel, chatModel, opts) =>
  ragEngine.createPipeline(name, embeddingModel, chatModel, opts)
);
handle('pipeline:delete', (_e, id) => ragEngine.deletePipeline(id));
handle('pipeline:get', (_e, id) => ragEngine.getPipeline(id));
handle('pipeline:rename', (_e, id, name) => ragEngine.renamePipeline(id, name));
handle('pipeline:update-settings', (_e, id, settings) =>
  ragEngine.updatePipelineSettings(id, settings)
);

// Document management
handle('pipeline:add-files', async (_e, pipelineId) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      {
        name: 'Documents',
        extensions: ['pdf', 'txt', 'md', 'docx', 'json', 'csv'],
      },
    ],
  });
  if (result.canceled) return { canceled: true };

  return ragEngine.ingestFiles(pipelineId, result.filePaths, (progress) => {
    mainWindow?.webContents.send('pipeline:ingest-progress', progress);
  });
});

handle('pipeline:add-file-paths', async (_e, pipelineId, filePaths) => {
  return ragEngine.ingestFiles(pipelineId, filePaths, (progress) => {
    mainWindow?.webContents.send('pipeline:ingest-progress', progress);
  });
});

handle('pipeline:documents', (_e, pipelineId) => ragEngine.listDocuments(pipelineId));
handle('pipeline:remove-document', (_e, pipelineId, docId) =>
  ragEngine.removeDocument(pipelineId, docId)
);

// Chat history
handle('chat:load', (_e, pipelineId) => ragEngine.loadChatHistory(pipelineId));
handle('chat:save', (_e, pipelineId, messages) => ragEngine.saveChatHistory(pipelineId, messages));
handle('chat:clear', (_e, pipelineId) => ragEngine.clearChatHistory(pipelineId));

// Streaming query
ipcMain.on('pipeline:query-stream', async (event, pipelineId, question, chatMessages) => {
  activeQueryAbort = new AbortController();
  const { signal } = activeQueryAbort;
  try {
    const result = await ragEngine.queryStream(pipelineId, question, chatMessages, (chunk) => {
      if (!signal.aborted) event.reply('pipeline:query-stream-chunk', chunk);
    }, signal);
    if (signal.aborted) {
      event.reply('pipeline:query-stream-done', []);
    } else {
      event.reply('pipeline:query-stream-done', result.sources);
    }
  } catch (err) {
    if (!signal.aborted) event.reply('pipeline:query-stream-error', err.message);
  } finally {
    activeQueryAbort = null;
  }
});

// Streaming deep query (deep thinking mode)
ipcMain.on('pipeline:query-stream-deep', async (event, pipelineId, question, chatMessages) => {
  activeQueryAbort = new AbortController();
  const { signal } = activeQueryAbort;
  try {
    const result = await ragEngine.queryStreamDeep(pipelineId, question, chatMessages, (chunk) => {
      if (!signal.aborted) event.reply('pipeline:query-stream-chunk', chunk);
    }, (thinking) => {
      if (!signal.aborted) event.reply('pipeline:query-stream-thinking', thinking);
    }, signal);
    if (signal.aborted) {
      event.reply('pipeline:query-stream-done', [], null);
    } else {
      event.reply('pipeline:query-stream-done', result.sources, result.subQueries);
    }
  } catch (err) {
    if (!signal.aborted) event.reply('pipeline:query-stream-error', err.message);
  } finally {
    activeQueryAbort = null;
  }
});

ipcMain.on('pipeline:query-abort', () => {
  if (activeQueryAbort) activeQueryAbort.abort();
});
