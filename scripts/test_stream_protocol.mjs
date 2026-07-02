/**
 * Minimal test harness for the streaming event protocol core.
 *
 * The repo has no vitest/jest; it runs plain Node `.mjs` scripts (see
 * parity_e2e.mjs). This mirrors that convention. Node v24 strips TypeScript
 * types natively, and src/services/stream-protocol.ts imports ONLY types, so we
 * can import the real module directly with no build step.
 *
 * Run: node scripts/test_stream_protocol.mjs
 */
import {
  isStreamEventFrame,
  isTerminalFrame,
  detectionBoxToTextBox,
  StreamAssembler,
} from '../src/services/stream-protocol.ts';

let passed = 0;
let failed = 0;

function ok(name, cond) {
  if (cond) {
    passed++;
    console.log(`  ok   - ${name}`);
  } else {
    failed++;
    console.error(`  FAIL - ${name}`);
  }
}

function eq(name, a, b) {
  ok(name, JSON.stringify(a) === JSON.stringify(b));
}

const SID = 'sess-123';
const det = (boxes) => ({ v: 1, type: 'detections', session_id: SID, boxes });
const tl = (index, translatedText, ocrText) => ({
  v: 1,
  type: 'tl',
  session_id: SID,
  index,
  translatedText,
  ...(ocrText !== undefined ? { ocrText } : {}),
});
const revise = (index, translatedText) => ({
  v: 1,
  type: 'revise',
  session_id: SID,
  index,
  translatedText,
});
const plate = (data) => ({ v: 1, type: 'plate', session_id: SID, data });
const done = (debug) => ({ v: 1, type: 'done', session_id: SID, debug });
const error = (msg) => ({ v: 1, type: 'error', session_id: SID, error: msg });

const box = (index, extra = {}) => ({
  index,
  minX: index * 10,
  minY: 0,
  maxX: index * 10 + 8,
  maxY: 8,
  ...extra,
});

/* -------------------- mode detection -------------------- */
console.log('mode detection');
ok('event frame is detected', isStreamEventFrame(det([box(0)])));
ok('tl frame is detected', isStreamEventFrame(tl(0, 'hi')));
ok('legacy response is NOT an event frame', !isStreamEventFrame({ images: [[]], success: true }));
ok('null is not a frame', !isStreamEventFrame(null));
ok('string is not a frame', !isStreamEventFrame('nope'));
ok('wrong version is not a frame', !isStreamEventFrame({ v: 2, type: 'tl' }));
ok('missing type is not a frame', !isStreamEventFrame({ v: 1 }));
ok('done is terminal', isTerminalFrame(done()));
ok('error is terminal', isTerminalFrame(error('x')));
ok('tl is not terminal', !isTerminalFrame(tl(0, 'a')));

/* -------------------- box hydration -------------------- */
console.log('detectionBoxToTextBox');
{
  const tb = detectionBoxToTextBox(box(2, { fontHeightPx: 24, bubbleRect: null }));
  eq('geometry mapped', [tb.minX, tb.maxX, tb.fontHeightPx], [20, 28, 24]);
  eq('empty text by default', [tb.translatedText, tb.ocrText], ['', '']);
}

/* -------------------- happy path, in order -------------------- */
console.log('assembler: in-order happy path');
{
  const a = new StreamAssembler();
  a.apply(det([box(0), box(1)]));
  a.apply(tl(0, 'HELLO', 'こん'));
  a.apply(tl(1, 'WORLD', 'せか'));
  a.apply(plate('PLATEDATA'));
  a.apply(done({ total_ms: 42 }));

  const res = a.toResponse();
  ok('session captured', a.session === SID);
  ok('isDone true', a.isDone === true);
  ok('no error', a.error === null);
  eq('two boxes in order', res.images[0].map((b) => b.translatedText), ['HELLO', 'WORLD']);
  eq('ocr captured', res.images[0].map((b) => b.ocrText), ['こん', 'せか']);
  eq('plate captured', res.inpainted_image_base64, ['PLATEDATA']);
  ok('debug carried', res.debug?.total_ms === 42);
  ok('success flag true', res.success === true);
  eq('images is array-of-arrays', [Array.isArray(res.images), Array.isArray(res.images[0])], [true, true]);
}

/* -------------------- idempotent tl (any order, repeats) -------------------- */
console.log('assembler: idempotent + out-of-order tl');
{
  const a = new StreamAssembler();
  a.apply(det([box(0), box(1)]));
  // tl for box 1 arrives before box 0; apply box0 twice (idempotent).
  a.apply(tl(1, 'SECOND'));
  a.apply(tl(0, 'FIRST'));
  a.apply(tl(0, 'FIRST')); // duplicate — must be stable
  const res = a.toResponse();
  eq('order preserved regardless of tl arrival', res.images[0].map((b) => b.translatedText), ['FIRST', 'SECOND']);
}

/* -------------------- tl BEFORE detections (buffered) -------------------- */
console.log('assembler: tl before detections is buffered + reconciled');
{
  const a = new StreamAssembler();
  a.apply(tl(0, 'EARLY', 'ocr0'));
  a.apply(det([box(0)]));
  const res = a.toResponse();
  eq('buffered text reconciled', [res.images[0][0].translatedText, res.images[0][0].ocrText], ['EARLY', 'ocr0']);
}

/* -------------------- revise supersedes tl -------------------- */
console.log('assembler: revise supersedes tl');
{
  const a = new StreamAssembler();
  a.apply(det([box(0)]));
  a.apply(tl(0, 'DRAFT', 'ocr'));
  a.apply(revise(0, 'FINAL'));
  const res = a.toResponse();
  eq('revise won, ocr preserved', [res.images[0][0].translatedText, res.images[0][0].ocrText], ['FINAL', 'ocr']);
}

/* -------------------- terminal error -------------------- */
console.log('assembler: terminal error');
{
  const a = new StreamAssembler();
  a.apply(det([box(0)]));
  a.apply(error('backend blew up'));
  const res = a.toResponse();
  ok('error captured', a.error === 'backend blew up');
  ok('isDone on error', a.isDone === true);
  ok('success false on error', res.success === false);
}

/* -------------------- summary -------------------- */
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
