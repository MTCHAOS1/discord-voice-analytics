"use strict";
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {dashboardView, formatDuration, mountDashboard} = require('../src/dashboard/assets/dashboard.js');

const fixture = {
  session_id: 'demo', demo: true, generated_at: '2026-10-27T12:00:00Z',
  participants: [
    {speaker: 'Alpha', estimated_speech_seconds: 60, segments: 3, content_word_count: 6},
    {speaker: 'Beta', estimated_speech_seconds: 20, segments: 2, content_word_count: 4}
  ],
  words: [{speaker: 'Alpha', word: 'jogo', count: 3}, {speaker: 'Beta', word: 'jogo', count: 2},
          {speaker: 'Beta', word: 'equipe', count: 2}]
};

test('soma palavras e mantem denominador da sessao ao filtrar', () => {
  const all = dashboardView(fixture);
  assert.equal(all.speech, 80);
  assert.equal(all.segments, 5);
  assert.equal(all.wordCount, 10);
  assert.deepEqual(all.words[0], {word: 'jogo', count: 5});
  const filtered = dashboardView(fixture, 'Beta', 1);
  assert.equal(filtered.participants.length, 1);
  assert.equal(filtered.participants[0].share, 25);
  assert.equal(filtered.speech, 20);
  assert.equal(filtered.words.length, 1);
  assert.equal(filtered.words[0].word, 'equipe');
});

test('zero, filtro vazio e duracoes longas', () => {
  const zero = {...fixture, participants: fixture.participants.map(p => ({...p, estimated_speech_seconds: 0}))};
  assert.equal(dashboardView(zero).participants[0].share, 0);
  assert.equal(dashboardView(fixture, 'ausente').speech, 0);
  assert.deepEqual(dashboardView(fixture, 'ausente').words, []);
  assert.equal(formatDuration(59.9), '1min 0s');
  assert.equal(formatDuration(3661), '1h 1min 1s');
});

// DOM minimo para verificar os eventos reais sem dependencias de navegador.
class Element {
  constructor() { this.children = []; this.style = {}; this.value = ''; this.events = {}; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute() {}
  addEventListener(event, handler) { this.events[event] = handler; }
}

test('interface atualiza indicadores, tabela e palavras com os filtros', () => {
  const elements = new Map();
  const doc = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  }, createElement() { return new Element(); }};
  doc.getElementById('limit').value = '10';
  mountDashboard(fixture, doc);
  assert.equal(doc.getElementById('people').textContent, '2');
  assert.equal(doc.getElementById('details').children.length, 2);
  doc.getElementById('speaker').value = 'Beta';
  doc.getElementById('speaker').events.change();
  assert.equal(doc.getElementById('people').textContent, '1');
  assert.equal(doc.getElementById('speech').textContent, '0min 20s');
  assert.equal(doc.getElementById('details').children.length, 1);
  doc.getElementById('limit').value = '1';
  doc.getElementById('limit').events.change();
  assert.equal(doc.getElementById('words').children.length, 1);
});

test('ranking respeita Top 5, 10 e 20 e restaura os totais ao limpar participante', () => {
  const elements = new Map();
  const doc = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  }, createElement() { return new Element(); }};
  const data = {...fixture, words: Array.from({length: 25}, (_, i) => ({
    speaker: i < 12 ? 'Alpha' : 'Beta', word: `termo${i}`, count: 25 - i
  }))};
  doc.getElementById('limit').value = '10';
  mountDashboard(data, doc);
  for (const limit of [5, 10, 20]) {
    doc.getElementById('limit').value = String(limit);
    doc.getElementById('limit').events.change();
    const rows = doc.getElementById('words').children;
    assert.equal(rows.length, limit);
    assert.equal(rows[0].children[0].children[0].textContent, 'termo0');
    assert.equal(rows[limit - 1].children[0].children[0].textContent, `termo${limit - 1}`);
  }
  doc.getElementById('speaker').value = 'Beta';
  doc.getElementById('speaker').events.change();
  assert.equal(doc.getElementById('words').children.length, 13);
  assert.equal(doc.getElementById('words').children[0].children[0].children[0].textContent, 'termo12');
  assert.equal(doc.getElementById('speech').textContent, '0min 20s');
  doc.getElementById('speaker').value = '';
  doc.getElementById('speaker').events.change();
  assert.equal(doc.getElementById('words').children.length, 20);
  assert.equal(doc.getElementById('people').textContent, '2');
  assert.equal(doc.getElementById('speech').textContent, '1min 20s');
  assert.equal(doc.getElementById('segments').textContent, '5');
  assert.equal(doc.getElementById('word-count').textContent, '10');
  assert.equal(doc.getElementById('details').children.length, 2);
});
