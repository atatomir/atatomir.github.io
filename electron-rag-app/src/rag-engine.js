const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const http = require('http');

// ── Helpers ────────────────────────────────────────────────────

const OLLAMA_BASE = 'http://127.0.0.1:11434';
const REQUEST_TIMEOUT = 120_000;

class OllamaError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'OllamaError';
    this.status = status;
  }
}

function ollamaFetch(urlPath, opts = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath, OLLAMA_BASE);
    const reqOpts = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: opts.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      timeout: opts.timeout || REQUEST_TIMEOUT,
    };
    const req = http.request(reqOpts, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (res.statusCode >= 400) {
            reject(new OllamaError(parsed.error || `HTTP ${res.statusCode}`, res.statusCode));
          } else {
            resolve({ status: res.statusCode, data: parsed });
          }
        } catch {
          if (res.statusCode >= 400) {
            reject(new OllamaError(`HTTP ${res.statusCode}: ${data}`, res.statusCode));
          } else {
            resolve({ status: res.statusCode, data });
          }
        }
      });
    });
    req.on('timeout', () => {
      req.destroy();
      reject(new OllamaError('Request timed out', 0));
    });
    req.on('error', (err) => {
      reject(new OllamaError(`Connection failed: ${err.message}`, 0));
    });
    if (opts.body) req.write(JSON.stringify(opts.body));
    req.end();
  });
}

function ollamaFetchStream(urlPath, body, onChunk) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath, OLLAMA_BASE);
    const reqOpts = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    };
    const req = http.request(reqOpts, (res) => {
      if (res.statusCode >= 400) {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            reject(new OllamaError(parsed.error || `HTTP ${res.statusCode}`, res.statusCode));
          } catch {
            reject(new OllamaError(`HTTP ${res.statusCode}`, res.statusCode));
          }
        });
        return;
      }
      let buffer = '';
      res.on('data', (chunk) => {
        buffer += chunk;
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            onChunk(JSON.parse(line));
          } catch {}
        }
      });
      res.on('end', () => {
        if (buffer.trim()) {
          try {
            onChunk(JSON.parse(buffer));
          } catch {}
        }
        resolve();
      });
    });
    req.on('error', (err) => {
      reject(new OllamaError(`Connection failed: ${err.message}`, 0));
    });
    req.write(JSON.stringify(body));
    req.end();
  });
}

// ── Vector Store (cosine similarity, no native deps) ──────────

class VectorStore {
  constructor(filePath) {
    this.filePath = filePath;
    this.vectors = [];
  }

  load() {
    if (fs.existsSync(this.filePath)) {
      try {
        this.vectors = JSON.parse(fs.readFileSync(this.filePath, 'utf-8'));
      } catch {
        this.vectors = [];
      }
    }
  }

  save() {
    const dir = path.dirname(this.filePath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.vectors));
  }

  add(id, embedding, metadata) {
    this.vectors.push({ id, embedding, metadata });
  }

  removeByDocId(docId) {
    this.vectors = this.vectors.filter((v) => v.metadata.docId !== docId);
  }

  count() {
    return this.vectors.length;
  }

  search(queryEmbedding, topK = 5, minScore = 0.0) {
    if (this.vectors.length === 0) return [];
    const scored = this.vectors.map((v) => ({
      ...v,
      score: cosineSimilarity(queryEmbedding, v.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.filter((s) => s.score >= minScore).slice(0, topK);
  }
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0,
    magA = 0,
    magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

// ── Document Parsers ──────────────────────────────────────────

async function parseFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const raw = fs.readFileSync(filePath);

  switch (ext) {
    case '.pdf': {
      const pdfParse = require('pdf-parse');
      const data = await pdfParse(raw);
      return data.text;
    }
    case '.docx': {
      const mammoth = require('mammoth');
      const result = await mammoth.extractRawText({ buffer: raw });
      return result.value;
    }
    case '.json': {
      const parsed = JSON.parse(raw.toString('utf-8'));
      return JSON.stringify(parsed, null, 2);
    }
    case '.csv': {
      const text = raw.toString('utf-8');
      const lines = text.split('\n').filter((l) => l.trim());
      if (lines.length === 0) return '';
      const headers = lines[0].split(',').map((h) => h.trim());
      const rows = lines.slice(1).map((line) => {
        const vals = line.split(',');
        return headers.map((h, i) => `${h}: ${(vals[i] || '').trim()}`).join(', ');
      });
      return rows.join('\n');
    }
    case '.txt':
    case '.md':
    default:
      return raw.toString('utf-8');
  }
}

// ── Chunking ──────────────────────────────────────────────────

function chunkText(text, chunkSize = 512, overlap = 64) {
  // Prefer splitting on sentence boundaries, falling back to word-level
  const sentences = text.match(/[^.!?\n]+[.!?\n]+|[^.!?\n]+$/g) || [text];
  const chunks = [];
  let current = [];
  let currentLen = 0;

  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/);
    if (currentLen + words.length > chunkSize && current.length > 0) {
      chunks.push(current.join(' '));
      // Keep overlap worth of words from the end
      const overlapWords = current.slice(-overlap);
      current = overlapWords;
      currentLen = overlapWords.length;
    }
    current.push(...words);
    currentLen += words.length;
  }

  if (current.length > 0) {
    const text = current.join(' ').trim();
    if (text) chunks.push(text);
  }

  // If no chunks were created (e.g., very short text), use the whole text
  if (chunks.length === 0 && text.trim()) {
    chunks.push(text.trim());
  }

  return chunks;
}

// ── RAG Engine ────────────────────────────────────────────────

class RagEngine {
  constructor(dataDir) {
    this.dataDir = path.join(dataDir, 'local-rag-data');
    this.pipelinesDir = path.join(this.dataDir, 'pipelines');
    this.metaPath = path.join(this.dataDir, 'pipelines.json');
    this.chatHistoryDir = path.join(this.dataDir, 'chat-history');
    this.pipelines = {};
    this.vectorStores = {};
  }

  async init() {
    fs.mkdirSync(this.pipelinesDir, { recursive: true });
    fs.mkdirSync(this.chatHistoryDir, { recursive: true });
    if (fs.existsSync(this.metaPath)) {
      try {
        this.pipelines = JSON.parse(fs.readFileSync(this.metaPath, 'utf-8'));
      } catch {
        this.pipelines = {};
      }
    }
    for (const id of Object.keys(this.pipelines)) {
      const vsPath = path.join(this.pipelinesDir, id, 'vectors.json');
      const vs = new VectorStore(vsPath);
      vs.load();
      this.vectorStores[id] = vs;
    }
  }

  _saveMeta() {
    fs.writeFileSync(this.metaPath, JSON.stringify(this.pipelines, null, 2));
  }

  // ── Chat History Persistence ────────────────────────────────

  saveChatHistory(pipelineId, messages) {
    const filePath = path.join(this.chatHistoryDir, `${pipelineId}.json`);
    fs.writeFileSync(filePath, JSON.stringify(messages, null, 2));
  }

  loadChatHistory(pipelineId) {
    const filePath = path.join(this.chatHistoryDir, `${pipelineId}.json`);
    if (fs.existsSync(filePath)) {
      try {
        return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      } catch {
        return [];
      }
    }
    return [];
  }

  clearChatHistory(pipelineId) {
    const filePath = path.join(this.chatHistoryDir, `${pipelineId}.json`);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  }

  // ── Ollama ────────────────────────────────────────────────

  async checkOllamaStatus() {
    try {
      const res = await ollamaFetch('/api/tags', { timeout: 3000 });
      return { running: res.status === 200 };
    } catch {
      return { running: false };
    }
  }

  async listModels() {
    try {
      const res = await ollamaFetch('/api/tags');
      return (res.data.models || []).map((m) => ({
        name: m.name,
        size: m.size,
        modified: m.modified_at,
        details: m.details || {},
      }));
    } catch {
      return [];
    }
  }

  async pullModel(name) {
    const res = await ollamaFetch('/api/pull', {
      method: 'POST',
      body: { name, stream: false },
      timeout: 600_000,
    });
    return res.data;
  }

  async deleteModel(name) {
    const res = await ollamaFetch('/api/delete', {
      method: 'DELETE',
      body: { name },
    });
    return res.data;
  }

  async _embed(model, text) {
    const res = await ollamaFetch('/api/embeddings', {
      method: 'POST',
      body: { model, prompt: text },
    });
    if (!res.data.embedding) {
      throw new OllamaError('No embedding returned. Is this an embedding model?', 0);
    }
    return res.data.embedding;
  }

  async _embedBatch(model, texts) {
    const embeddings = [];
    // Ollama doesn't support batch embeddings natively, so we parallelize with concurrency limit
    const CONCURRENCY = 4;
    for (let i = 0; i < texts.length; i += CONCURRENCY) {
      const batch = texts.slice(i, i + CONCURRENCY);
      const results = await Promise.all(batch.map((t) => this._embed(model, t)));
      embeddings.push(...results);
    }
    return embeddings;
  }

  async _chatStream(model, messages, onToken) {
    await ollamaFetchStream('/api/chat', { model, messages, stream: true }, (obj) => {
      if (obj.message?.content) {
        onToken(obj.message.content);
      }
    });
  }

  // ── Pipeline CRUD ─────────────────────────────────────────

  listPipelines() {
    return Object.entries(this.pipelines).map(([id, p]) => ({
      id,
      ...p,
      chunkCount: this.vectorStores[id]?.count() || 0,
    }));
  }

  createPipeline(name, embeddingModel, chatModel, opts = {}) {
    const id = crypto.randomUUID();
    const pipelineDir = path.join(this.pipelinesDir, id);
    fs.mkdirSync(pipelineDir, { recursive: true });

    this.pipelines[id] = {
      name,
      embeddingModel,
      chatModel,
      chunkSize: opts.chunkSize || 512,
      chunkOverlap: opts.chunkOverlap || 64,
      topK: opts.topK || 5,
      systemPrompt:
        opts.systemPrompt ||
        'You are a helpful assistant that answers questions based on the provided context. Use ONLY the context below to answer. If the context doesn\'t contain the answer, say so clearly. Cite sources using [n] notation.',
      documents: [],
      createdAt: new Date().toISOString(),
    };
    this._saveMeta();

    const vs = new VectorStore(path.join(pipelineDir, 'vectors.json'));
    this.vectorStores[id] = vs;
    vs.save();

    return { id, ...this.pipelines[id] };
  }

  deletePipeline(id) {
    if (!this.pipelines[id]) return;
    const pipelineDir = path.join(this.pipelinesDir, id);
    fs.rmSync(pipelineDir, { recursive: true, force: true });
    this.clearChatHistory(id);
    delete this.pipelines[id];
    delete this.vectorStores[id];
    this._saveMeta();
  }

  getPipeline(id) {
    if (!this.pipelines[id]) return null;
    return {
      id,
      ...this.pipelines[id],
      chunkCount: this.vectorStores[id]?.count() || 0,
    };
  }

  renamePipeline(id, newName) {
    if (!this.pipelines[id]) return null;
    this.pipelines[id].name = newName;
    this._saveMeta();
    return this.getPipeline(id);
  }

  updatePipelineSettings(id, settings) {
    if (!this.pipelines[id]) return null;
    const allowed = ['chatModel', 'embeddingModel', 'chunkSize', 'chunkOverlap', 'topK', 'systemPrompt'];
    for (const key of allowed) {
      if (settings[key] !== undefined) {
        this.pipelines[id][key] = settings[key];
      }
    }
    this._saveMeta();
    return this.getPipeline(id);
  }

  // ── Document Ingestion ────────────────────────────────────

  async ingestFiles(pipelineId, filePaths, onProgress) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) throw new Error('Pipeline not found');

    const vs = this.vectorStores[pipelineId];
    const results = [];
    const totalFiles = filePaths.length;

    for (let fileIdx = 0; fileIdx < totalFiles; fileIdx++) {
      const filePath = filePaths[fileIdx];
      const docId = crypto.randomUUID();
      const fileName = path.basename(filePath);

      if (onProgress) {
        onProgress({
          type: 'file-start',
          fileName,
          fileIndex: fileIdx,
          totalFiles,
        });
      }

      const text = await parseFile(filePath);
      const chunks = chunkText(text, pipeline.chunkSize, pipeline.chunkOverlap);

      // Embed in batches with progress
      const BATCH = 8;
      for (let i = 0; i < chunks.length; i += BATCH) {
        const batch = chunks.slice(i, i + BATCH);
        const embeddings = await this._embedBatch(pipeline.embeddingModel, batch);
        for (let j = 0; j < batch.length; j++) {
          vs.add(`${docId}_${i + j}`, embeddings[j], {
            docId,
            fileName,
            chunkIndex: i + j,
            text: batch[j],
          });
        }
        if (onProgress) {
          onProgress({
            type: 'embedding-progress',
            fileName,
            chunksProcessed: Math.min(i + BATCH, chunks.length),
            totalChunks: chunks.length,
          });
        }
      }

      pipeline.documents.push({
        id: docId,
        fileName,
        filePath,
        chunks: chunks.length,
        addedAt: new Date().toISOString(),
      });

      results.push({ docId, fileName, chunks: chunks.length });

      if (onProgress) {
        onProgress({
          type: 'file-done',
          fileName,
          fileIndex: fileIdx,
          totalFiles,
          chunks: chunks.length,
        });
      }
    }

    vs.save();
    this._saveMeta();
    return results;
  }

  listDocuments(pipelineId) {
    return this.pipelines[pipelineId]?.documents || [];
  }

  removeDocument(pipelineId, docId) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) return;
    pipeline.documents = pipeline.documents.filter((d) => d.id !== docId);
    this.vectorStores[pipelineId].removeByDocId(docId);
    this.vectorStores[pipelineId].save();
    this._saveMeta();
  }

  // ── Query ─────────────────────────────────────────────────

  async queryStream(pipelineId, question, chatMessages, onToken) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) throw new Error('Pipeline not found');

    const vs = this.vectorStores[pipelineId];

    if (vs.count() === 0) {
      throw new Error('No documents in this pipeline. Add some documents first.');
    }

    const queryEmbedding = await this._embed(pipeline.embeddingModel, question);
    const results = vs.search(queryEmbedding, pipeline.topK || 5, 0.1);

    if (results.length === 0) {
      throw new Error('No relevant chunks found for your query.');
    }

    // Deduplicate near-identical chunks
    const seen = new Set();
    const dedupedResults = results.filter((r) => {
      const key = r.metadata.text.substring(0, 100);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    const context = dedupedResults
      .map((r, i) => `[${i + 1}] (${r.metadata.fileName}) ${r.metadata.text}`)
      .join('\n\n');

    const systemPrompt = `${pipeline.systemPrompt}\n\nContext:\n${context}`;

    // Include recent chat history for conversational context (last 6 messages)
    const recentHistory = (chatMessages || []).slice(-6);

    const messages = [
      { role: 'system', content: systemPrompt },
      ...recentHistory,
      { role: 'user', content: question },
    ];

    await this._chatStream(pipeline.chatModel, messages, onToken);

    return {
      sources: dedupedResults.map((r) => ({
        fileName: r.metadata.fileName,
        chunkIndex: r.metadata.chunkIndex,
        text: r.metadata.text.length > 200 ? r.metadata.text.substring(0, 200) + '...' : r.metadata.text,
        score: r.score,
      })),
    };
  }
}

module.exports = { RagEngine, VectorStore, cosineSimilarity, chunkText, parseFile, OllamaError };
