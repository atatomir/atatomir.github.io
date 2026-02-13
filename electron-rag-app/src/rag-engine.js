const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const http = require('http');

// ── Helpers ────────────────────────────────────────────────────

function ollamaFetch(urlPath, opts = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath, 'http://127.0.0.1:11434');
    const reqOpts = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: opts.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
    };
    const req = http.request(reqOpts, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, data });
        }
      });
    });
    req.on('error', reject);
    if (opts.body) req.write(JSON.stringify(opts.body));
    req.end();
  });
}

function ollamaFetchStream(urlPath, body, onChunk) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath, 'http://127.0.0.1:11434');
    const reqOpts = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    };
    const req = http.request(reqOpts, (res) => {
      let buffer = '';
      res.on('data', (chunk) => {
        buffer += chunk;
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const obj = JSON.parse(line);
            onChunk(obj);
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
    req.on('error', reject);
    req.write(JSON.stringify(body));
    req.end();
  });
}

// ── Simple Vector Store (cosine similarity, no native deps) ───

class VectorStore {
  constructor(filePath) {
    this.filePath = filePath;
    this.vectors = []; // { id, embedding, metadata }
  }

  load() {
    if (fs.existsSync(this.filePath)) {
      this.vectors = JSON.parse(fs.readFileSync(this.filePath, 'utf-8'));
    }
  }

  save() {
    fs.writeFileSync(this.filePath, JSON.stringify(this.vectors));
  }

  add(id, embedding, metadata) {
    this.vectors.push({ id, embedding, metadata });
  }

  removeByDocId(docId) {
    this.vectors = this.vectors.filter((v) => v.metadata.docId !== docId);
  }

  search(queryEmbedding, topK = 5) {
    const scored = this.vectors.map((v) => ({
      ...v,
      score: cosineSimilarity(queryEmbedding, v.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-10);
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
      return JSON.stringify(JSON.parse(raw.toString('utf-8')), null, 2);
    }
    case '.csv':
    case '.txt':
    case '.md':
    default:
      return raw.toString('utf-8');
  }
}

// ── Chunking ──────────────────────────────────────────────────

function chunkText(text, chunkSize = 512, overlap = 64) {
  const words = text.split(/\s+/);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    if (chunk.trim()) chunks.push(chunk);
    i += chunkSize - overlap;
  }
  return chunks;
}

// ── RAG Engine ────────────────────────────────────────────────

class RagEngine {
  constructor(dataDir) {
    this.dataDir = path.join(dataDir, 'local-rag-data');
    this.pipelinesDir = path.join(this.dataDir, 'pipelines');
    this.metaPath = path.join(this.dataDir, 'pipelines.json');
    this.pipelines = {};
    this.vectorStores = {};
  }

  async init() {
    fs.mkdirSync(this.pipelinesDir, { recursive: true });
    if (fs.existsSync(this.metaPath)) {
      this.pipelines = JSON.parse(fs.readFileSync(this.metaPath, 'utf-8'));
    }
    // Load vector stores for existing pipelines
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

  // ── Ollama ────────────────────────────────────────────────

  async checkOllamaStatus() {
    try {
      const res = await ollamaFetch('/api/tags');
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
      }));
    } catch {
      return [];
    }
  }

  async pullModel(name) {
    const res = await ollamaFetch('/api/pull', {
      method: 'POST',
      body: { name, stream: false },
    });
    return res.data;
  }

  async _embed(model, text) {
    const res = await ollamaFetch('/api/embeddings', {
      method: 'POST',
      body: { model, prompt: text },
    });
    return res.data.embedding;
  }

  async _chat(model, messages, stream = false) {
    if (stream) throw new Error('Use _chatStream for streaming');
    const res = await ollamaFetch('/api/chat', {
      method: 'POST',
      body: { model, messages, stream: false },
    });
    return res.data.message?.content || '';
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
    return Object.entries(this.pipelines).map(([id, p]) => ({ id, ...p }));
  }

  createPipeline(name, embeddingModel, chatModel) {
    const id = crypto.randomUUID();
    const pipelineDir = path.join(this.pipelinesDir, id);
    fs.mkdirSync(pipelineDir, { recursive: true });

    this.pipelines[id] = {
      name,
      embeddingModel,
      chatModel,
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
    delete this.pipelines[id];
    delete this.vectorStores[id];
    this._saveMeta();
  }

  getPipeline(id) {
    if (!this.pipelines[id]) return null;
    return { id, ...this.pipelines[id] };
  }

  // ── Document Ingestion ────────────────────────────────────

  async ingestFiles(pipelineId, filePaths) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) throw new Error('Pipeline not found');

    const vs = this.vectorStores[pipelineId];
    const results = [];

    for (const filePath of filePaths) {
      const docId = crypto.randomUUID();
      const fileName = path.basename(filePath);

      // Parse
      const text = await parseFile(filePath);

      // Chunk
      const chunks = chunkText(text);

      // Embed each chunk
      for (let i = 0; i < chunks.length; i++) {
        const embedding = await this._embed(pipeline.embeddingModel, chunks[i]);
        vs.add(`${docId}_${i}`, embedding, {
          docId,
          fileName,
          chunkIndex: i,
          text: chunks[i],
        });
      }

      pipeline.documents.push({
        id: docId,
        fileName,
        chunks: chunks.length,
        addedAt: new Date().toISOString(),
      });

      results.push({ docId, fileName, chunks: chunks.length });
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

  async query(pipelineId, question) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) throw new Error('Pipeline not found');

    const vs = this.vectorStores[pipelineId];
    const queryEmbedding = await this._embed(pipeline.embeddingModel, question);
    const results = vs.search(queryEmbedding, 5);

    const context = results
      .map((r, i) => `[${i + 1}] (${r.metadata.fileName}) ${r.metadata.text}`)
      .join('\n\n');

    const systemPrompt = `You are a helpful assistant that answers questions based on the provided context. Use ONLY the context below to answer. If the context doesn't contain the answer, say so clearly. Cite sources using [n] notation.

Context:
${context}`;

    const answer = await this._chat(pipeline.chatModel, [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: question },
    ]);

    return {
      answer,
      sources: results.map((r) => ({
        fileName: r.metadata.fileName,
        text: r.metadata.text.substring(0, 200) + '...',
        score: r.score,
      })),
    };
  }

  async queryStream(pipelineId, question, onToken) {
    const pipeline = this.pipelines[pipelineId];
    if (!pipeline) throw new Error('Pipeline not found');

    const vs = this.vectorStores[pipelineId];
    const queryEmbedding = await this._embed(pipeline.embeddingModel, question);
    const results = vs.search(queryEmbedding, 5);

    const context = results
      .map((r, i) => `[${i + 1}] (${r.metadata.fileName}) ${r.metadata.text}`)
      .join('\n\n');

    const systemPrompt = `You are a helpful assistant that answers questions based on the provided context. Use ONLY the context below to answer. If the context doesn't contain the answer, say so clearly. Cite sources using [n] notation.

Context:
${context}`;

    await this._chatStream(
      pipeline.chatModel,
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question },
      ],
      onToken
    );

    return {
      sources: results.map((r) => ({
        fileName: r.metadata.fileName,
        text: r.metadata.text.substring(0, 200) + '...',
        score: r.score,
      })),
    };
  }
}

module.exports = { RagEngine };
