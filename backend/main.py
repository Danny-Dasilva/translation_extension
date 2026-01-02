#!/usr/bin/env python3
"""Run the Manga Translation API server."""

import os
import uvicorn

if __name__ == "__main__":
    # Increase Uvicorn's flow control buffer for faster large body reads
    # Default 64KB causes 31 pause/resume cycles for 2MB uploads
    # Must be patched BEFORE uvicorn.run() is called
    from uvicorn.protocols.http import flow_control
    flow_control.HIGH_WATER_LIMIT = 262144  # 256KB (4x larger)

    # Reload mode disabled by default for optimal performance with llama-cpp
    # Enable with RELOAD=true environment variable for development
    use_reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=use_reload,
        log_level="info",
        # Performance optimizations for large body handling
        http="httptools",  # C-accelerated HTTP parser (faster than h11)
        loop="uvloop",     # 2-4x faster event loop than asyncio
    )
