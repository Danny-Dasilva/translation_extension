"""Unit tests for the flag-for-finetune endpoint (POST /flag).

Users flag poor translations from the extension; the backend persists the
ORIGINAL source image + metadata locally as a fine-tune dataset seed. The
write must be asynchronous (BackgroundTasks / asyncio.to_thread) so the
endpoint returns immediately without blocking on disk I/O.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image


def _tiny_png_b64(as_data_url: bool = True) -> str:
    """A 2x2 red PNG, base64-encoded (optionally as a data URL)."""
    img = Image.new("RGB", (2, 2), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = base64.b64encode(buf.getvalue()).decode("ascii")
    if as_data_url:
        return "data:image/png;base64," + raw
    return raw


@pytest.fixture()
def flag_client(tmp_path, monkeypatch):
    """A TestClient whose flagged_dir points at a fresh temp directory."""
    # Point the configurable storage dir at a temp dir BEFORE importing the
    # router (the router reads settings.flagged_dir at request time, so a
    # monkeypatch on the live settings object is sufficient).
    from app.config import settings

    flagged_dir = tmp_path / "flagged"
    monkeypatch.setattr(settings, "flagged_dir", str(flagged_dir), raising=False)

    # Import lazily so the router module picks up the patched settings.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import flag as flag_router

    app = FastAPI()
    app.include_router(flag_router.router)
    client = TestClient(app)
    return client, flagged_dir


def test_flag_persists_image_and_jsonl(flag_client):
    client, flagged_dir = flag_client

    payload = {
        "image_base64": _tiny_png_b64(as_data_url=True),
        "page_url": "https://example.com/manga/ch1/p3",
        "target_language": "English",
        "boxes": [
            {
                "ocr_text": "こんにちは",
                "translated_text": "Hello",
                "minX": 10,
                "minY": 20,
                "maxX": 110,
                "maxY": 60,
            }
        ],
        "note": "mistranslated greeting",
        "reason": "wrong_translation",
    }

    resp = client.post("/flag", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["id"]
    assert body["image_path"]

    # The background task runs on the TestClient request lifecycle; by the time
    # the context manager / call returns, the file + record must exist.
    image_path = Path(flagged_dir) / Path(body["image_path"]).name
    assert image_path.exists(), f"image not written: {image_path}"
    assert image_path.stat().st_size > 0

    jsonl_path = Path(flagged_dir) / "flagged.jsonl"
    assert jsonl_path.exists(), "flagged.jsonl not written"
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == body["id"]
    assert record["page_url"] == "https://example.com/manga/ch1/p3"
    assert record["target_language"] == "English"
    assert record["note"] == "mistranslated greeting"
    assert record["reason"] == "wrong_translation"
    assert record["boxes"][0]["ocr_text"] == "こんにちは"
    assert record["boxes"][0]["translated_text"] == "Hello"
    assert record["boxes"][0]["minX"] == 10
    # image_path stored in the record is relative (the seed-dataset path)
    assert record["image_path"]
    assert not Path(record["image_path"]).is_absolute()


def test_flag_accepts_raw_base64_without_data_url(flag_client):
    client, flagged_dir = flag_client
    payload = {
        "image_base64": _tiny_png_b64(as_data_url=False),
        "boxes": [],
    }
    resp = client.post("/flag", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    image_path = Path(flagged_dir) / Path(body["image_path"]).name
    assert image_path.exists()


def test_flag_rejects_invalid_image(flag_client):
    client, _ = flag_client
    payload = {"image_base64": "not-valid-base64-$$$", "boxes": []}
    resp = client.post("/flag", json=payload)
    assert resp.status_code == 400


def test_flag_appends_multiple_records(flag_client):
    client, flagged_dir = flag_client
    img = _tiny_png_b64()
    for i in range(3):
        resp = client.post("/flag", json={"image_base64": img, "boxes": [], "note": f"n{i}"})
        assert resp.status_code == 200
    jsonl_path = Path(flagged_dir) / "flagged.jsonl"
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
