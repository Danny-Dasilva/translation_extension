"""Unit tests for vertical-AR-by-default routing in ParseqOCRService.

The failure mode being fixed: the fast non-AR (NAR) PARSeq decode duplicates /
substitutes adjacent kana on dense VERTICAL crops at FALSELY-HIGH confidence
(身代わり -> 身身わわ at conf 0.92), so the confidence-gated AR retry never fires
on the worst cases. The fix routes tall/narrow (h/w >= aspect) crops to the AR
model UP FRONT, by geometry, independent of confidence.

These tests drive the routing logic with a fake service that stubs both the NAR
and AR session.run, so no real model / GPU is needed. The NAR path is made to
emit a known garble for vertical crops; the AR path emits the correct read. We
assert that:
  * vertical crops (h/w >= aspect) are decoded by the AR session,
  * horizontal crops stay on the fast NAR session,
  * results are 1:1 with inputs and stitched back in original order,
  * a config flag (default on) toggles the behavior,
  * the aspect predicate matches the existing _maybe_rotate_vertical threshold.
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.services.parseq_ocr_service import ParseqOCRService


def _logits_for(token_ids):
    """Build (1, L, C=4) high-conf logits for a token sequence + EOS.

    Charset is "abc": head idx 0=EOS, 1='a', 2='b', 3='c'.
    """
    rows = []
    for tok in token_ids:
        row = [0.0, 0.0, 0.0, 0.0]
        row[tok] = 10.0
        rows.append(row)
    rows.append([10.0, 0.0, 0.0, 0.0])  # EOS
    return np.array([rows], dtype=np.float32)


class _FakeVerticalAR(ParseqOCRService):
    """Bypass __init__/ONNX; stub NAR + AR session.run keyed by crop marker.

    The crop's first pixel encodes a crop id. NAR maps:
      id 0 (horizontal clean) -> 'a'   (correct)
      id 1 (vertical garble)  -> 'bb'  (the duplicated-kana garble signature)
    AR maps everything to the CORRECT read:
      id 0 -> 'a',  id 1 -> 'bc'  (recovered, no dup)
    """

    def __init__(self, vertical_ar_default: bool = True, vertical_ar_aspect: float = 1.5):
        self.charset = "abc"
        self.eos_id = 0
        self.head_dim = 4
        self._itos = ["[E]"] + list(self.charset)
        self.img_h, self.img_w = 128, 512
        self.mean = np.full((1, 3, 1, 1), 0.5, dtype=np.float32)
        self.std = np.full((1, 3, 1, 1), 0.5, dtype=np.float32)
        self._input_name = "images"
        self._input_np_dtype = np.float32
        # hybrid / AR config
        self._ar_model_path = "models/parseq_manga_ep60_AR_dynbatch.onnx"
        self._ar_session = None
        self._ar_input_name = "images"
        self._ar_input_np_dtype = np.float32
        self.hybrid_enabled = False  # low-conf retry OFF; vertical routing is separate
        self.hybrid_conf_threshold = 0.65
        self.ar_retry_count = 0
        # NEW: vertical-AR-by-default knobs
        self.vertical_ar_default = bool(vertical_ar_default)
        self.vertical_ar_aspect = float(vertical_ar_aspect)
        self.vertical_ar_count = 0
        # call counters for assertions
        self.nar_crop_ids: list[int] = []
        self.ar_crop_ids: list[int] = []
        self._fake_ar_loaded = False

    def _preprocess(self, crops):
        # Encode crop id (first pixel) in the returned tensor for dispatch.
        batch = np.zeros((len(crops), 3, self.img_h, self.img_w), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i, 0, 0, 0] = float(c[0, 0, 0])
        return batch

    def _run_sync(self, batch):
        out = []
        for row in batch:
            cid = int(round(float(row[0, 0, 0])))
            self.nar_crop_ids.append(cid)
            if cid == 0:
                out.append(_logits_for([1])[0])       # 'a' correct
            else:
                out.append(_logits_for([2, 2])[0])    # 'bb' GARBLE (dup)
        return _pad(out)

    def _ensure_ar_session(self):
        self._fake_ar_loaded = True
        return True

    def _run_ar_sync(self, batch):
        out = []
        for row in batch:
            cid = int(round(float(row[0, 0, 0])))
            self.ar_crop_ids.append(cid)
            if cid == 0:
                out.append(_logits_for([1])[0])       # 'a'
            else:
                out.append(_logits_for([2, 3])[0])    # 'bc' RECOVERED (no dup)
        return _pad(out)


def _pad(rows):
    L = max(o.shape[0] for o in rows)
    C = rows[0].shape[1]
    padded = np.zeros((len(rows), L, C), dtype=np.float32)
    for i, o in enumerate(rows):
        padded[i, : o.shape[0]] = o
        padded[i, o.shape[0] :, 0] = 10.0  # EOS pad
    return padded


def _h_crop(cid: int) -> np.ndarray:
    """Horizontal crop (wide): h < aspect * w. id encoded in first pixel."""
    c = np.zeros((20, 100, 3), dtype=np.uint8)
    c[0, 0, 0] = cid
    return c


def _v_crop(cid: int) -> np.ndarray:
    """Vertical crop (tall/narrow): h >= aspect * w. id encoded in first pixel."""
    c = np.zeros((300, 30, 3), dtype=np.uint8)
    c[0, 0, 0] = cid
    return c


def test_vertical_crop_routed_to_ar():
    f = _FakeVerticalAR(vertical_ar_default=True)
    crops = [_v_crop(1)]  # tall/narrow vertical garble crop
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert len(tc) == 1
    # AR recovered it to 'bc' (no dup) instead of NAR garble 'bb'.
    assert tc[0][0] == "bc"
    assert 1 in f.ar_crop_ids       # AR saw the vertical crop
    assert 1 not in f.nar_crop_ids  # NAR did NOT process it


def test_horizontal_crop_stays_on_nar():
    f = _FakeVerticalAR(vertical_ar_default=True)
    crops = [_h_crop(0)]  # wide horizontal clean crop
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert tc[0][0] == "a"
    assert 0 in f.nar_crop_ids      # NAR processed the horizontal crop
    assert 0 not in f.ar_crop_ids   # AR never touched it


def test_mixed_batch_partitions_and_preserves_order():
    f = _FakeVerticalAR(vertical_ar_default=True)
    # interleaved: horizontal, vertical, horizontal, vertical
    crops = [_h_crop(0), _v_crop(1), _h_crop(0), _v_crop(1)]
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert [t for t, _ in tc] == ["a", "bc", "a", "bc"]
    # horizontals -> NAR, verticals -> AR
    assert sorted(f.nar_crop_ids) == [0, 0]
    assert sorted(f.ar_crop_ids) == [1, 1]


def test_flag_off_keeps_vertical_on_nar():
    f = _FakeVerticalAR(vertical_ar_default=False)
    crops = [_v_crop(1)]
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    # routing disabled -> NAR garble 'bb' survives, AR never loaded
    assert tc[0][0] == "bb"
    assert f.ar_crop_ids == []
    assert f._fake_ar_loaded is False


def test_aspect_threshold_matches_rotate_vertical():
    # The routing predicate must use the SAME tall/narrow test as
    # _maybe_rotate_vertical (h > aspect * w) so "rotated" == "routed".
    f = _FakeVerticalAR(vertical_ar_default=True, vertical_ar_aspect=1.5)
    # h/w = 1.4 -> NOT vertical (below 1.5); h/w = 1.6 -> vertical.
    just_horizontal = np.zeros((140, 100, 3), dtype=np.uint8)
    just_horizontal[0, 0, 0] = 0
    just_vertical = np.zeros((160, 100, 3), dtype=np.uint8)
    just_vertical[0, 0, 0] = 1
    asyncio.run(f._recognize_batch_with_conf([just_horizontal, just_vertical]))
    assert 0 in f.nar_crop_ids   # 1.4 aspect -> NAR
    assert 1 in f.ar_crop_ids    # 1.6 aspect -> AR


def test_is_vertical_predicate():
    f = _FakeVerticalAR(vertical_ar_aspect=1.5)
    assert f._is_vertical_crop(np.zeros((300, 30, 3), dtype=np.uint8)) is True
    assert f._is_vertical_crop(np.zeros((20, 100, 3), dtype=np.uint8)) is False
    # exactly at threshold (h == 1.5 * w) is NOT vertical (strict >, matches rotate)
    assert f._is_vertical_crop(np.zeros((150, 100, 3), dtype=np.uint8)) is False
    assert f._is_vertical_crop(np.zeros((151, 100, 3), dtype=np.uint8)) is True
