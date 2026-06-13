"""Data pipeline for JP->EN manga translator fine-tune.

All loaders emit parquet files matching the schema in `unify_schema.py`:
    {jp: str, en: str, src: str, register_tag: str, gold_flag: bool}
"""
