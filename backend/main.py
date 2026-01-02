#!/usr/bin/env python3
"""Run the Manga Translation API server."""

import os
import uvicorn

if __name__ == "__main__":
    # Reload mode disabled by default for optimal performance with llama-cpp
    # Enable with RELOAD=true environment variable for development
    use_reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=use_reload,
    )
