"""Unit tests for the confidence-gated HYBRID OCR (non-AR -> AR-retry).

The batch recognize path runs the fast non-AR model over all crops, then
re-OCRs ONLY the low-confidence crops with the higher-quality AR model and
replaces those results. We drive the logic with a fake service that stubs the
ONNX session.run for both the non-AR and the lazily-loaded AR session, so no
real model/GPU is needed.
"""
from __future__ import annotations

import asyncio

import numpy as np

from app.services.parseq_ocr_service import ParseqOCRService


def _logits_for(seq_conf):
    """Build (1, L, C=3) logits encoding a token sequence with target conf.

    ``seq_conf`` is a list of (token_id, strong) tuples; strong=True -> high
    max-prob (~1.0), strong=False -> flat/low max-prob. Terminated by EOS.
    Charset is "ab": head idx 0=EOS, 1='a', 2='b'.
    """
    rows = []
    for tok, strong in seq_conf:
        row = [0.0, 0.0, 0.0]
        if strong:
            row[tok] = 10.0
        else:
            # near-flat: argmax still tok but tiny margin -> low max-prob
            row[tok] = 0.20
            for j in range(3):
                if j != tok:
                    row[j] = 0.15
        rows.append(row)
    rows.append([10.0, 0.0, 0.0])  # EOS
    return np.array([rows], dtype=np.float32)


class _FakeHybrid(ParseqOCRService):
    """Bypass __init__/ONNX; stub non-AR and AR session.run.

    non-AR: crop 0 -> 'a' strong (high conf), crop 1 -> 'b' flat (low conf).
    AR:     crop (the low-conf one) -> 'ab' strong (high conf, recovered).
    """

    def __init__(self):  # noqa: D401 - test shim
        self.charset = "ab"
        self.eos_id = 0
        self.head_dim = 3
        self._itos = ["[E]"] + list(self.charset)
        self.img_h, self.img_w = 128, 512
        self.mean = np.full((1, 3, 1, 1), 0.5, dtype=np.float32)
        self.std = np.full((1, 3, 1, 1), 0.5, dtype=np.float32)
        self._input_name = "images"
        self._input_np_dtype = np.float32
        # hybrid config
        self._ar_model_path = "models/parseq_manga_ep60_AR_dynbatch.onnx"
        self._ar_session = None
        self._ar_input_name = "images"
        self._ar_input_np_dtype = np.float32
        self.hybrid_enabled = True
        self.hybrid_conf_threshold = 0.65
        self.ar_retry_count = 0  # populated by impl

        # marker so the impl knows the AR session is "loaded"
        self._fake_ar_loaded = False

    # The non-AR forward: encode per-crop logits keyed by a tag we stash on the
    # preprocessed batch. We instead override _run_sync to return per-row logits
    # based on a side-channel set in _preprocess (the crops' first pixel value).
    def _preprocess(self, crops):
        # encode crop "id" in the returned tensor so _run_sync can dispatch
        batch = np.zeros((len(crops), 3, self.img_h, self.img_w), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i, 0, 0, 0] = float(c[0, 0, 0])  # crop id marker
        return batch

    def _run_sync(self, batch):
        # non-AR: crop id 0 -> high-conf 'a'; crop id 1 -> low-conf 'b'
        out = []
        for row in batch:
            cid = int(round(float(row[0, 0, 0])))
            if cid == 0:
                lg = _logits_for([(1, True)])  # 'a' strong
            else:
                lg = _logits_for([(2, False)])  # 'b' flat (low conf)
            out.append(lg[0])
        # pad to equal length
        L = max(o.shape[0] for o in out)
        padded = np.zeros((len(out), L, 3), dtype=np.float32)
        for i, o in enumerate(out):
            padded[i, : o.shape[0]] = o
            padded[i, o.shape[0] :, 0] = 10.0  # EOS pad
        return padded

    def _ensure_ar_session(self):
        self._fake_ar_loaded = True
        return True

    def _run_ar_sync(self, batch):
        # AR recovers everything to high-conf 'ab'
        out = []
        for _row in batch:
            out.append(_logits_for([(1, True), (2, True)])[0])
        L = max(o.shape[0] for o in out)
        padded = np.zeros((len(out), L, 3), dtype=np.float32)
        for i, o in enumerate(out):
            padded[i, : o.shape[0]] = o
            padded[i, o.shape[0] :, 0] = 10.0
        return padded


def _make_crops():
    c0 = np.zeros((4, 4, 3), dtype=np.uint8)  # id 0 -> high conf
    c1 = np.ones((4, 4, 3), dtype=np.uint8)  # id 1 -> low conf
    return [c0, c1]


def test_low_conf_crop_retried_with_ar():
    f = _FakeHybrid()
    crops = _make_crops()
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert len(tc) == 2
    # crop 0 untouched (high conf non-AR 'a')
    assert tc[0][0] == "a"
    assert tc[0][1] > 0.9
    # crop 1 was low-conf -> AR-replaced with high-conf 'ab'
    assert tc[1][0] == "ab"
    assert tc[1][1] > 0.9
    # exactly one crop AR-retried
    assert f.ar_retry_count == 1


def test_ar_disabled_keeps_nonar():
    f = _FakeHybrid()
    f.hybrid_enabled = False
    crops = _make_crops()
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert tc[1][0] == "b"  # not replaced
    assert tc[1][1] < 0.65
    assert f.ar_retry_count == 0


def test_no_low_conf_no_ar_call():
    f = _FakeHybrid()
    # both crops high conf (id 0)
    crops = [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)]
    tc = asyncio.run(f._recognize_batch_with_conf(crops))
    assert all(t == "a" for t, _ in tc)
    assert f.ar_retry_count == 0
    assert f._fake_ar_loaded is False  # AR never loaded
