"""Curated-bitext mining pipeline: page-aligned (JP, EN) page images -> curated
JP->EN training rows in the v11 page-context parquet schema.

Modules:
  geometry      bbox + normalized-position + reading-order helpers
  align         cross-page JP<->EN bubble alignment (Hungarian on normalized pos)
  curate        precision-favoring quality filters + per-row quality score
  format_rows   byte-exact v11 page-context prompt + [prompt,en,src,register_tag,gold_flag]
  ocr_adapters  JP = CTD+PARSeq (CPU-capable); EN = VLM (deferred, needs GPU)
  pipeline      align_and_curate engine + per-image orchestration
  run_gallery   CLI: mine over the manifest into local parquet shards
  smoke_cpu     CPU-only real-OCR smoke test on Ikenie source pages
  validate_ikenie  known-answer validation (alignment P/R + sample parquet)
"""
