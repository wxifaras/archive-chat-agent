#!/usr/bin/env python3
"""
Startup script for the FastAPI Archive Chat Agent API
"""
import uvicorn
from main import app
from core.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower() if hasattr(settings, 'LOG_LEVEL') else 'info'
    )
