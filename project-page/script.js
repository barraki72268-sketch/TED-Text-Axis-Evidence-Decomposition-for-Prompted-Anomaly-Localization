'use strict';

const tabs = [...document.querySelectorAll('[data-example]')];
const exampleImage = document.getElementById('example-image');
const exampleZoom = document.getElementById('example-zoom');
const panel = document.getElementById('example-panel');
const exampleStatus = document.getElementById('example-status');
let currentExample = 1;

function showExample(number, moveFocus = false) {
  currentExample = ((number - 1 + tabs.length) % tabs.length) + 1;
  tabs.forEach((tab, index) => {
    const selected = index + 1 === currentExample;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  const path = `assets/qualitative-${currentExample}.webp`;
  exampleImage.src = path;
  exampleImage.alt = `Example ${currentExample}: original input with ground truth, FAPrompt host anomaly map, and C-TED anomaly map.`;
  exampleZoom.dataset.zoom = path;
  exampleZoom.dataset.caption = `Qualitative example ${currentExample}. Original TED paper, row ${currentExample} of the qualitative comparison.`;
  panel.setAttribute('aria-labelledby', `example-tab-${currentExample}`);
  exampleStatus.textContent = `Example ${currentExample} of ${tabs.length}`;
  if (moveFocus) tabs[currentExample - 1].focus();
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => showExample(Number(tab.dataset.example)));
  tab.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      showExample(currentExample + (event.key === 'ArrowRight' ? 1 : -1), true);
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      showExample(event.key === 'Home' ? 1 : tabs.length, true);
    }
  });
});
document.getElementById('previous-example').addEventListener('click', () => showExample(currentExample - 1));
document.getElementById('next-example').addEventListener('click', () => showExample(currentExample + 1));

const dialog = document.getElementById('image-dialog');
const dialogImage = document.getElementById('dialog-image');
const dialogCaption = document.getElementById('dialog-caption');
let zoomTrigger;
document.querySelectorAll('[data-zoom]').forEach(button => {
  button.addEventListener('click', () => {
    zoomTrigger = button;
    dialogImage.src = button.dataset.zoom;
    dialogImage.alt = button.querySelector('img').alt;
    dialogCaption.textContent = button.dataset.caption;
    dialog.showModal();
  });
});
document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => zoomTrigger?.focus());
dialog.addEventListener('click', event => { if (event.target === dialog) {
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
}});

document.getElementById('copy-citation').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  const text = document.getElementById('bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'BibTeX selected. Copy the selected text.';
  }
});
