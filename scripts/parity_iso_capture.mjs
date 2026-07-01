#!/usr/bin/env node
/**
 * parity_iso_capture.mjs — Capture ONE canonical /translate response per page.
 *
 * This is the SINGLE SOURCE OF TRUTH fed to BOTH renderers (extension canvas
 * path A + backend PIL path B), isolating the renderer from inference variance.
 *
 * Flow: read the bench source webp, base64 it (data:image/...), POST to
 * http://127.0.0.1:8001/translate exactly ONCE, and persist the raw JSON
 * (TextBox[] + inpainted_image_base64 plate) to
 * backend/.bench/_parity/iso_<page>/response.json.
 *
 * Usage: node scripts/parity_iso_capture.mjs <page>   (044 | 030)
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const PAGE = process.argv[2] || '044';
const API = process.env.PARITY_API || 'http://127.0.0.1:8001';
const SRC = path.join(ROOT, `backend/.bench/full_pipeline_v2/588828_mesu2_insp/${PAGE}/01_source.webp`);
const OUT_DIR = path.join(ROOT, `backend/.bench/_parity/iso_${PAGE}`);
fs.mkdirSync(OUT_DIR, { recursive: true });

if (!fs.existsSync(SRC)) { console.error('FATAL: source missing', SRC); process.exit(2); }

const bytes = fs.readFileSync(SRC);
// The extension compresses to JPEG before sending; the backend re-decodes either
// way. We send the source as webp data-url (backend PIL/cv2 decodes webp). This
// is the SAME image both paths render text on top of (the plate is what the
// renderers actually draw on, so source codec is not load-bearing for parity).
const dataUrl = `data:image/webp;base64,${bytes.toString('base64')}`;

const body = JSON.stringify({ base64Images: [dataUrl], targetLanguage: 'English' });

const t0 = Date.now();
const res = await fetch(`${API}/translate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body,
});
if (!res.ok) {
  console.error('FATAL: /translate returned', res.status, await res.text().catch(() => ''));
  process.exit(1);
}
const json = await res.json();
const dt = Date.now() - t0;

const boxes = (json.images && json.images[0]) || [];
const plate = (json.inpainted_image_base64 && json.inpainted_image_base64[0]) || null;

// Persist the raw response (single source of truth for BOTH renderers).
fs.writeFileSync(path.join(OUT_DIR, 'response.json'), JSON.stringify(json));
// Also a compact summary for humans.
const summary = {
  page: PAGE,
  elapsed_ms: dt,
  num_boxes: boxes.length,
  has_plate: !!plate,
  plate_prefix: plate ? plate.slice(0, 40) : null,
  box_fields_sample: boxes[0] ? Object.keys(boxes[0]) : [],
  translations: boxes.map((b) => b.translatedText),
  bubble_matched: boxes.filter((b) => b.bubbleRect).length,
};
fs.writeFileSync(path.join(OUT_DIR, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
