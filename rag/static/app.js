const form = document.getElementById('ask-form');
const question = document.getElementById('question');
const result = document.getElementById('result');
const answer = document.getElementById('answer');
const sources = document.getElementById('sources');
const error = document.getElementById('error');
const button = document.getElementById('ask-button');

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function renderAnswer(text, citations) {
  answer.replaceChildren();
  const byId = new Map(citations.map(c => [c.id, c]));
  text.split(/\n\n+/).forEach(paragraph => {
    const p = node('p');
    paragraph.split(/(\[E\d+\])/).forEach(piece => {
      const id = piece.slice(1, -1);
      if (byId.has(id) && piece === `[${id}]`) {
        const a = node('a', 'cite', piece);
        a.href = `#source-${id}`;
        p.append(a);
      } else p.append(document.createTextNode(piece));
    });
    answer.append(p);
  });
}
function renderSources(citations) {
  sources.replaceChildren();
  if (!citations.length) return;
  sources.append(node('h3', '', 'Cited passages'));
  citations.forEach(c => {
    const section = node('section', 'source'); section.id = `source-${c.id}`;
    section.append(node('p', 'source-title', `${c.id} · ${c.book}`));
    section.append(node('p', 'source-meta', `${c.chapter || 'Chapter unspecified'} · PDF page ${c.pdf_page ?? 'unknown'} · ${c.source_id}`));
    section.append(node('p', 'source-excerpt', c.excerpt));
    const a = node('a', '', 'Open cited PDF page ↗');
    a.href = `/api/pdf/${encodeURIComponent(c.book)}#page=${c.pdf_page || 1}`;
    a.target = '_blank'; a.rel = 'noopener noreferrer';
    section.append(a);
    const detail = node('a', '', 'View extracted passage');
    detail.href = `/source/${encodeURIComponent(c.chunk_id)}`;
    detail.style.marginLeft = '1rem';
    section.append(detail); sources.append(section);
  });
}
async function health() {
  try {
    const response = await fetch('/api/health'); const data = await response.json();
    document.getElementById('index-state').textContent = data.status === 'ready' ? `${data.books} books indexed` : data.status === 'index_outdated' ? 'Rebuild index' : 'Index needed';
  } catch { document.getElementById('index-state').textContent = 'Service unavailable'; }
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  error.classList.add('hidden'); result.classList.remove('hidden');
  document.getElementById('status').textContent = 'Searching';
  answer.replaceChildren(node('div', 'loading'), node('div', 'loading'));
  sources.replaceChildren(); button.disabled = true;
  try {
    const response = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:question.value.trim(), audience:document.querySelector('input[name="audience"]:checked').value})});
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || 'Request failed');
    document.getElementById('status').textContent = data.status.replaceAll('_', ' ');
    renderAnswer(data.answer, data.citations); renderSources(data.citations);
  } catch (e) {
    result.classList.add('hidden'); error.textContent = `Could not complete the search: ${e.message}`; error.classList.remove('hidden');
  } finally { button.disabled = false; }
});
health();
