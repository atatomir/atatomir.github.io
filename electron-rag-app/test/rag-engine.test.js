const { describe, it, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const fs = require('fs');
const os = require('os');

const { VectorStore, cosineSimilarity, chunkText, RagEngine, DOC_PRESETS, parseCSVRow } = require('../src/rag-engine');

// ── cosineSimilarity ──────────────────────────────────────────

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const v = [1, 2, 3, 4, 5];
    const sim = cosineSimilarity(v, v);
    assert.ok(Math.abs(sim - 1) < 1e-6, `Expected ~1, got ${sim}`);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    const sim = cosineSimilarity(a, b);
    assert.ok(Math.abs(sim) < 1e-6, `Expected ~0, got ${sim}`);
  });

  it('returns -1 for opposite vectors', () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    const sim = cosineSimilarity(a, b);
    assert.ok(Math.abs(sim + 1) < 1e-6, `Expected ~-1, got ${sim}`);
  });

  it('handles zero vectors gracefully', () => {
    const sim = cosineSimilarity([0, 0, 0], [1, 2, 3]);
    assert.equal(sim, 0);
  });

  it('handles null/undefined', () => {
    assert.equal(cosineSimilarity(null, [1, 2]), 0);
    assert.equal(cosineSimilarity([1, 2], undefined), 0);
  });

  it('handles mismatched lengths', () => {
    assert.equal(cosineSimilarity([1, 2], [1, 2, 3]), 0);
  });

  it('computes known similarity', () => {
    const a = [1, 2, 3];
    const b = [4, 5, 6];
    // dot = 32, |a| = sqrt(14), |b| = sqrt(77)
    const expected = 32 / (Math.sqrt(14) * Math.sqrt(77));
    const sim = cosineSimilarity(a, b);
    assert.ok(Math.abs(sim - expected) < 1e-6);
  });
});

// ── chunkText ─────────────────────────────────────────────────

describe('chunkText', () => {
  it('returns single chunk for short text', () => {
    const chunks = chunkText('Hello world', 100, 10);
    assert.equal(chunks.length, 1);
    assert.equal(chunks[0], 'Hello world');
  });

  it('splits long text into multiple chunks', () => {
    // Use sentences so recursive text splitter can split on '. '
    const sentences = Array.from({ length: 100 }, (_, i) => `This is sentence number ${i}.`);
    const text = sentences.join(' ');
    // Character-based chunk size — each sentence is ~30 chars, use 200 chars per chunk
    const chunks = chunkText(text, 200, 20);
    assert.ok(chunks.length > 1, `Expected >1 chunks, got ${chunks.length}`);
  });

  it('preserves content - no words lost', () => {
    const words = Array.from({ length: 50 }, (_, i) => `w${i}`);
    const text = words.join('. ') + '.';
    // Character-based: text is ~200 chars, use 50-char chunks
    const chunks = chunkText(text, 50, 10);
    // Every original word should appear in at least one chunk
    for (const word of words) {
      const found = chunks.some((c) => c.includes(word));
      assert.ok(found, `Word "${word}" not found in any chunk`);
    }
  });

  it('handles empty text', () => {
    const chunks = chunkText('', 100, 10);
    assert.equal(chunks.length, 0);
  });

  it('handles single-word text', () => {
    const chunks = chunkText('hello', 100, 10);
    assert.equal(chunks.length, 1);
    assert.equal(chunks[0], 'hello');
  });

  it('respects sentence boundaries', () => {
    const text =
      'First sentence here. Second sentence here. Third sentence here. Fourth sentence here.';
    // Character-based: text is ~88 chars, use 45-char chunks
    const chunks = chunkText(text, 45, 5);
    // Each chunk should tend to start at a sentence boundary
    assert.ok(chunks.length >= 2, `Expected >=2 chunks, got ${chunks.length}`);
  });

  it('handles text with no punctuation', () => {
    // Without sentence boundaries, the recursive splitter falls through to space/char separators
    const text = Array.from({ length: 100 }, (_, i) => `word${i}`).join(' ');
    // Character-based: text is ~600 chars, use 100-char chunks
    const chunks = chunkText(text, 100, 20);
    assert.ok(chunks.length >= 1, `Expected >=1 chunks, got ${chunks.length}`);
    // All words should be present across all chunks
    for (let i = 0; i < 100; i++) {
      assert.ok(chunks.some((c) => c.includes(`word${i}`)), `word${i} missing`);
    }
  });
});

// ── VectorStore ───────────────────────────────────────────────

describe('VectorStore', () => {
  let tmpDir;
  let vsPath;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-test-'));
    vsPath = path.join(tmpDir, 'vectors.json');
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('starts empty', () => {
    const vs = new VectorStore(vsPath);
    assert.equal(vs.count(), 0);
    assert.deepEqual(vs.search([1, 0, 0]), []);
  });

  it('adds vectors and searches', () => {
    const vs = new VectorStore(vsPath);
    vs.add('a', [1, 0, 0], { text: 'right' });
    vs.add('b', [0, 1, 0], { text: 'up' });
    vs.add('c', [0, 0, 1], { text: 'forward' });

    assert.equal(vs.count(), 3);

    const results = vs.search([1, 0.1, 0], 2);
    assert.equal(results.length, 2);
    assert.equal(results[0].metadata.text, 'right');
  });

  it('saves and loads', () => {
    const vs1 = new VectorStore(vsPath);
    vs1.add('x', [1, 2, 3], { text: 'test' });
    vs1.save();

    const vs2 = new VectorStore(vsPath);
    vs2.load();
    assert.equal(vs2.count(), 1);
    assert.equal(vs2.vectors[0].metadata.text, 'test');
  });

  it('removes by docId', () => {
    const vs = new VectorStore(path.join(tmpDir, 'vs2.json'));
    vs.add('a1', [1, 0], { docId: 'doc1', text: 'a' });
    vs.add('a2', [0, 1], { docId: 'doc1', text: 'b' });
    vs.add('b1', [1, 1], { docId: 'doc2', text: 'c' });

    assert.equal(vs.count(), 3);
    vs.removeByDocId('doc1');
    assert.equal(vs.count(), 1);
    assert.equal(vs.vectors[0].metadata.docId, 'doc2');
  });

  it('respects minScore filter', () => {
    const vs = new VectorStore(path.join(tmpDir, 'vs3.json'));
    vs.add('a', [1, 0, 0], { text: 'close' });
    vs.add('b', [0, 1, 0], { text: 'orthogonal' });

    const results = vs.search([1, 0, 0], 5, 0.5);
    assert.equal(results.length, 1);
    assert.equal(results[0].metadata.text, 'close');
  });

  it('handles loading non-existent file', () => {
    const vs = new VectorStore(path.join(tmpDir, 'nonexistent.json'));
    vs.load();
    assert.equal(vs.count(), 0);
  });

  it('handles loading corrupt file', () => {
    const corruptPath = path.join(tmpDir, 'corrupt.json');
    fs.writeFileSync(corruptPath, 'not valid json{{{');
    const vs = new VectorStore(corruptPath);
    vs.load();
    assert.equal(vs.count(), 0);
  });

  it('getByDocId returns matching vectors', () => {
    const vs = new VectorStore(path.join(tmpDir, 'vs4.json'));
    vs.add('a1', [1, 0], { docId: 'doc1', text: 'a' });
    vs.add('a2', [0, 1], { docId: 'doc1', text: 'b' });
    vs.add('b1', [1, 1], { docId: 'doc2', text: 'c' });

    const doc1Vecs = vs.getByDocId('doc1');
    assert.equal(doc1Vecs.length, 2);
    assert.ok(doc1Vecs.every((v) => v.metadata.docId === 'doc1'));
  });

  it('toJSON returns vectors array', () => {
    const vs = new VectorStore(path.join(tmpDir, 'vs5.json'));
    vs.add('x', [1, 2], { text: 'hello' });
    const json = vs.toJSON();
    assert.ok(Array.isArray(json));
    assert.equal(json.length, 1);
    assert.equal(json[0].metadata.text, 'hello');
  });

  it('loadFromData replaces vectors', () => {
    const vs = new VectorStore(path.join(tmpDir, 'vs6.json'));
    vs.add('old', [1, 0], { text: 'old' });
    assert.equal(vs.count(), 1);
    vs.loadFromData([{ id: 'new', embedding: [0, 1], metadata: { text: 'new' } }]);
    assert.equal(vs.count(), 1);
    assert.equal(vs.vectors[0].metadata.text, 'new');
  });
});

// ── RagEngine (CRUD - no Ollama needed) ───────────────────────

describe('RagEngine (offline CRUD)', () => {
  let tmpDir;
  let engine;

  before(async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-engine-test-'));
    engine = new RagEngine(tmpDir);
    await engine.init();
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('starts with no pipelines', () => {
    const list = engine.listPipelines();
    assert.equal(list.length, 0);
  });

  it('creates a pipeline with defaults from general preset', () => {
    const p = engine.createPipeline('Test Pipeline', 'nomic-embed-text', 'llama3.2', {
      chunkSize: 256,
      topK: 3,
    });
    assert.ok(p.id);
    assert.equal(p.name, 'Test Pipeline');
    assert.equal(p.embeddingModel, 'nomic-embed-text');
    assert.equal(p.chatModel, 'llama3.2');
    assert.equal(p.chunkSize, 256);
    assert.equal(p.topK, 3);
    // New fields should have defaults from general preset
    assert.equal(p.temperature, DOC_PRESETS.general.temperature);
    assert.equal(p.minScore, DOC_PRESETS.general.minScore);
    assert.equal(p.contextWindow, DOC_PRESETS.general.contextWindow);
    assert.equal(p.preset, 'general');
  });

  it('creates a pipeline with a specific preset', () => {
    const p = engine.createPipeline('Code Pipeline', 'nomic-embed-text', 'llama3.2', {
      preset: 'code',
    });
    assert.equal(p.chunkSize, DOC_PRESETS.code.chunkSize);
    assert.equal(p.chunkOverlap, DOC_PRESETS.code.chunkOverlap);
    assert.equal(p.topK, DOC_PRESETS.code.topK);
    assert.equal(p.temperature, DOC_PRESETS.code.temperature);
    assert.equal(p.minScore, DOC_PRESETS.code.minScore);
    assert.equal(p.preset, 'code');
    // Cleanup
    engine.deletePipeline(p.id);
  });

  it('lists pipelines', () => {
    const list = engine.listPipelines();
    assert.equal(list.length, 1);
    assert.equal(list[0].name, 'Test Pipeline');
  });

  it('gets pipeline by id', () => {
    const list = engine.listPipelines();
    const p = engine.getPipeline(list[0].id);
    assert.equal(p.name, 'Test Pipeline');
    assert.equal(p.chunkCount, 0);
  });

  it('renames pipeline', () => {
    const list = engine.listPipelines();
    const p = engine.renamePipeline(list[0].id, 'Renamed Pipeline');
    assert.equal(p.name, 'Renamed Pipeline');
  });

  it('updates pipeline settings', () => {
    const list = engine.listPipelines();
    const p = engine.updatePipelineSettings(list[0].id, {
      topK: 10,
      chunkSize: 1024,
      temperature: 0.5,
      minScore: 0.4,
      contextWindow: 8,
    });
    assert.equal(p.topK, 10);
    assert.equal(p.chunkSize, 1024);
    assert.equal(p.temperature, 0.5);
    assert.equal(p.minScore, 0.4);
    assert.equal(p.contextWindow, 8);
  });

  it('rejects unknown settings', () => {
    const list = engine.listPipelines();
    engine.updatePipelineSettings(list[0].id, { hackerField: 'evil' });
    const p = engine.getPipeline(list[0].id);
    assert.equal(p.hackerField, undefined);
  });

  it('chat history CRUD', () => {
    const list = engine.listPipelines();
    const id = list[0].id;

    engine.saveChatHistory(id, [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
    ]);

    const loaded = engine.loadChatHistory(id);
    assert.equal(loaded.length, 2);
    assert.equal(loaded[0].content, 'hello');

    engine.clearChatHistory(id);
    const cleared = engine.loadChatHistory(id);
    assert.equal(cleared.length, 0);
  });

  it('returns null for nonexistent pipeline', () => {
    assert.equal(engine.getPipeline('nonexistent-id'), null);
  });

  it('deletes pipeline', () => {
    const list = engine.listPipelines();
    engine.deletePipeline(list[0].id);
    assert.equal(engine.listPipelines().length, 0);
  });

  it('persists across reloads', async () => {
    engine.createPipeline('Persisted', 'emb', 'chat');
    const engine2 = new RagEngine(tmpDir);
    await engine2.init();
    const list = engine2.listPipelines();
    assert.equal(list.length, 1);
    assert.equal(list[0].name, 'Persisted');
    // Cleanup
    engine2.deletePipeline(list[0].id);
  });

  it('creates pipeline with provider options', () => {
    const p = engine.createPipeline('OpenAI Pipeline', 'text-embedding-3-small', 'gpt-4o-mini', {
      embeddingProvider: 'openai',
      chatProvider: 'openai',
    });
    assert.equal(p.embeddingProvider, 'openai');
    assert.equal(p.chatProvider, 'openai');
    assert.equal(p.embeddingModel, 'text-embedding-3-small');
    assert.equal(p.chatModel, 'gpt-4o-mini');
    engine.deletePipeline(p.id);
  });

  it('saves and loads config', () => {
    engine.saveConfig({ openaiApiKey: 'test-key', openaiBaseUrl: 'https://custom.api.com' });
    const config = engine.getConfig();
    assert.equal(config.openaiApiKey, 'test-key');
    assert.equal(config.openaiBaseUrl, 'https://custom.api.com');
    // Clean up
    engine.saveConfig({ openaiApiKey: '', openaiBaseUrl: 'https://api.openai.com' });
  });

  it('exports and imports pipeline', () => {
    const p = engine.createPipeline('Export Test', 'emb', 'chat', { chunkSize: 500 });

    const exported = engine.exportPipeline(p.id);
    assert.equal(exported.version, 2);
    assert.equal(exported.pipeline.name, 'Export Test');
    assert.ok(exported.exportedAt);
    assert.ok(Array.isArray(exported.vectors));
    assert.ok(Array.isArray(exported.chatHistory));

    const imported = engine.importPipeline(exported);
    assert.ok(imported.id);
    assert.notEqual(imported.id, p.id); // New ID
    assert.equal(imported.name, 'Export Test');
    assert.equal(imported.chunkSize, 500);

    // Cleanup
    engine.deletePipeline(p.id);
    engine.deletePipeline(imported.id);
  });

  it('getDocumentChunks returns empty for pipeline with no docs', () => {
    const p = engine.createPipeline('Empty', 'emb', 'chat');
    const chunks = engine.getDocumentChunks(p.id, 'nonexistent-doc');
    assert.deepEqual(chunks, []);
    engine.deletePipeline(p.id);
  });
});

// ── parseFile ─────────────────────────────────────────────────

describe('parseFile', () => {
  const { parseFile } = require('../src/rag-engine');
  let tmpDir;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'parse-test-'));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('parses .txt files', async () => {
    const filePath = path.join(tmpDir, 'test.txt');
    fs.writeFileSync(filePath, 'Hello from text file');
    const text = await parseFile(filePath);
    assert.equal(text, 'Hello from text file');
  });

  it('parses .md files', async () => {
    const filePath = path.join(tmpDir, 'test.md');
    fs.writeFileSync(filePath, '# Title\n\nSome markdown content');
    const text = await parseFile(filePath);
    assert.ok(text.includes('Title'));
    assert.ok(text.includes('markdown content'));
  });

  it('parses .json files', async () => {
    const filePath = path.join(tmpDir, 'test.json');
    fs.writeFileSync(filePath, JSON.stringify({ key: 'value', num: 42 }));
    const text = await parseFile(filePath);
    const parsed = JSON.parse(text);
    assert.equal(parsed.key, 'value');
  });

  it('parses .csv files into readable row format', async () => {
    const filePath = path.join(tmpDir, 'test.csv');
    fs.writeFileSync(filePath, 'Name,Age,City\nAlice,30,NYC\nBob,25,LA');
    const text = await parseFile(filePath);
    assert.ok(text.includes('Name: Alice'));
    assert.ok(text.includes('Age: 30'));
    assert.ok(text.includes('City: NYC'));
    assert.ok(text.includes('Name: Bob'));
    assert.ok(text.includes('Row 1:'), 'Should include row labels');
    assert.ok(text.includes('Row 2:'), 'Should include row labels');
  });

  it('parses .csv with quoted fields correctly', async () => {
    const filePath = path.join(tmpDir, 'quoted.csv');
    fs.writeFileSync(filePath, 'Name,Location,Note\n"Smith, John","New York, NY","He said ""hello"""\nJane,LA,Simple');
    const text = await parseFile(filePath);
    assert.ok(text.includes('Name: Smith, John'), `Should preserve comma in quotes. Got: ${text}`);
    assert.ok(text.includes('Location: New York, NY'), 'Should preserve comma in quoted location');
    assert.ok(text.includes('Note: He said "hello"'), 'Should handle escaped quotes');
    assert.ok(text.includes('Name: Jane'));
  });

  it('handles unknown extensions as text', async () => {
    const filePath = path.join(tmpDir, 'test.xyz');
    fs.writeFileSync(filePath, 'unknown format content');
    const text = await parseFile(filePath);
    assert.equal(text, 'unknown format content');
  });
});

// ── DOC_PRESETS ───────────────────────────────────────────────

describe('DOC_PRESETS', () => {
  it('has all expected presets', () => {
    const expected = ['general', 'technical', 'legal', 'code', 'research', 'csv'];
    for (const key of expected) {
      assert.ok(DOC_PRESETS[key], `Missing preset: ${key}`);
    }
  });

  it('each preset has required fields', () => {
    const requiredFields = ['label', 'description', 'chunkSize', 'chunkOverlap', 'topK', 'temperature', 'minScore', 'contextWindow', 'systemPrompt'];
    for (const [key, preset] of Object.entries(DOC_PRESETS)) {
      for (const field of requiredFields) {
        assert.ok(preset[field] !== undefined, `Preset "${key}" missing field "${field}"`);
      }
    }
  });

  it('all presets have context-only instructions in system prompt', () => {
    for (const [key, preset] of Object.entries(DOC_PRESETS)) {
      assert.ok(
        preset.systemPrompt.includes('STRICTLY') || preset.systemPrompt.includes('ONLY'),
        `Preset "${key}" system prompt should enforce context-only answers`
      );
    }
  });

  it('temperature values are in valid range', () => {
    for (const [key, preset] of Object.entries(DOC_PRESETS)) {
      assert.ok(preset.temperature >= 0 && preset.temperature <= 2, `Preset "${key}" temperature out of range: ${preset.temperature}`);
    }
  });
});

// ── parseCSVRow ───────────────────────────────────────────────

describe('parseCSVRow', () => {
  it('parses simple CSV row', () => {
    assert.deepEqual(parseCSVRow('a,b,c'), ['a', 'b', 'c']);
  });

  it('handles quoted fields with commas', () => {
    assert.deepEqual(parseCSVRow('"Smith, John",30,"New York, NY"'), ['Smith, John', '30', 'New York, NY']);
  });

  it('handles escaped quotes in quoted fields', () => {
    assert.deepEqual(parseCSVRow('"He said ""hello""",plain'), ['He said "hello"', 'plain']);
  });

  it('handles empty fields', () => {
    assert.deepEqual(parseCSVRow('a,,c'), ['a', '', 'c']);
  });

  it('handles single field', () => {
    assert.deepEqual(parseCSVRow('hello'), ['hello']);
  });
});

// ── CJK chunking ─────────────────────────────────────────────

describe('chunkText (CJK)', () => {
  it('splits on CJK sentence delimiters', () => {
    const text = 'First sentence here. 这是第一句。这是第二句。This is third.';
    // Character-based: text is ~55 chars, use 25-char chunks
    const chunks = chunkText(text, 25, 5);
    assert.ok(chunks.length >= 2, `Expected >=2 chunks, got ${chunks.length}`);
  });

  it('handles Japanese sentence endings', () => {
    const text = 'これは最初の文です。これは二番目の文です。三番目の文です。';
    // Character-based: text is ~30 chars, use 15-char chunks
    const chunks = chunkText(text, 15, 3);
    assert.ok(chunks.length >= 1, `Should produce chunks for Japanese text`);
  });
});
