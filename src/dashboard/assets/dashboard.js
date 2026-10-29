"use strict";

function dashboardView(data, selected = "", limit = 10) {
  const total = data.participants.reduce((sum, p) => sum + p.estimated_speech_seconds, 0);
  const participants = data.participants.filter(p => !selected || p.speaker === selected)
    .map(p => ({...p, share: total > 0 ? p.estimated_speech_seconds / total * 100 : 0}))
    .sort((a, b) => b.estimated_speech_seconds - a.estimated_speech_seconds || a.speaker.localeCompare(b.speaker));
  const frequencies = new Map();
  for (const row of data.words) {
    if (!selected || row.speaker === selected) frequencies.set(row.word, (frequencies.get(row.word) || 0) + row.count);
  }
  return {
    participants,
    speech: participants.reduce((sum, p) => sum + p.estimated_speech_seconds, 0),
    segments: participants.reduce((sum, p) => sum + p.segments, 0),
    wordCount: participants.reduce((sum, p) => sum + p.content_word_count, 0),
    words: [...frequencies].map(([word, count]) => ({word, count}))
      .sort((a, b) => b.count - a.count || a.word.localeCompare(b.word)).slice(0, limit)
  };
}

function formatDuration(seconds) {
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor(rounded % 3600 / 60);
  const rest = rounded % 60;
  return hours ? `${hours}h ${minutes}min ${rest}s` : `${minutes}min ${rest}s`;
}

function mountDashboard(data, doc) {
  const byId = id => doc.getElementById(id);
  const integer = number => number.toLocaleString("pt-BR");
  byId("session").textContent = data.session_id;
  byId("quality").textContent = data.demo ? "DEMONSTRAÇÃO" : `QUALIDADE ${data.quality_status}`;
  byId("notice").textContent = data.demo
    ? "Demonstração com dados fictícios. Nenhuma gravação foi utilizada."
    : data.quality_status === "WARN"
      ? "Relatório com alertas, incluído após confirmação de revisão. Confira a qualidade antes de interpretar."
      : "Dados passaram pelas verificações locais de consistência. A precisão da transcrição exige revisão humana.";
  byId("generated").textContent = `Gerado em ${new Date(data.generated_at).toLocaleString("pt-BR")}`
    + (data.validated_at ? ` · Validação em ${new Date(data.validated_at).toLocaleString("pt-BR")}` : "");
  for (const person of data.participants) {
    const option = doc.createElement("option");
    option.value = person.speaker;
    option.textContent = person.speaker;
    byId("speaker").appendChild(option);
  }

  function bar(container, label, value, percentage) {
    const row = doc.createElement("div"); row.className = "bar-row";
    const heading = doc.createElement("div"); heading.className = "bar-label";
    for (const text of [label, value]) {
      const span = doc.createElement("span"); span.textContent = text; heading.appendChild(span);
    }
    const track = doc.createElement("div"); track.className = "track"; track.setAttribute("aria-hidden", "true");
    const fill = doc.createElement("div"); fill.className = "fill";
    fill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    track.appendChild(fill); row.appendChild(heading); row.appendChild(track); container.appendChild(row);
  }

  function render() {
    const view = dashboardView(data, byId("speaker").value, Number(byId("limit").value));
    byId("speech").textContent = formatDuration(view.speech);
    byId("people").textContent = integer(view.participants.length);
    byId("segments").textContent = integer(view.segments);
    byId("word-count").textContent = integer(view.wordCount);
    byId("participation").replaceChildren(); byId("words").replaceChildren(); byId("details").replaceChildren();
    for (const p of view.participants) {
      const share = `${p.share.toLocaleString("pt-BR", {maximumFractionDigits: 1})}%`;
      bar(byId("participation"), p.speaker, `${formatDuration(p.estimated_speech_seconds)} · ${share}`, p.share);
      const row = doc.createElement("tr");
      for (const value of [p.speaker, formatDuration(p.estimated_speech_seconds), share, integer(p.segments), integer(p.content_word_count)]) {
        const cell = doc.createElement("td"); cell.textContent = value; row.appendChild(cell);
      }
      byId("details").appendChild(row);
    }
    const max = view.words.length ? view.words[0].count : 0;
    for (const word of view.words) bar(byId("words"), word.word, integer(word.count), max ? word.count / max * 100 : 0);
    if (!view.words.length) {
      const empty = doc.createElement("p"); empty.className = "empty";
      empty.textContent = "Nenhuma palavra de conteúdo para este filtro."; byId("words").appendChild(empty);
    }
  }
  byId("speaker").addEventListener("change", render);
  byId("limit").addEventListener("change", render);
  render();
}

if (typeof module !== "undefined" && module.exports) module.exports = {dashboardView, formatDuration, mountDashboard};
if (typeof document !== "undefined") mountDashboard(JSON.parse(document.getElementById("dashboard-data").textContent), document);
