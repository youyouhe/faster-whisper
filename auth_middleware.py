#!/usr/bin/env python3
"""
API Key Authentication Middleware for ASR Services
Provides simple API key based authentication for HTTP endpoints
"""

import os
from typing import Optional
from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader
import logging

logger = logging.getLogger(__name__)

class APIKeyAuth:
    """API Key authentication middleware"""

    def __init__(self, api_key: Optional[str] = None, header_name: str = "X-API-Key"):
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.header_name = header_name
        self.api_key_header = APIKeyHeader(name=header_name, auto_error=False)

        if not self.api_key:
            logger.warning("No API key configured! Services will be open to all requests.")
        else:
            logger.info(f"API key authentication enabled with header: {header_name}")

    def verify_api_key(self, request: Request) -> bool:
        """Verify API key from request headers"""
        if not self.api_key:
            # No API key configured - allow all requests (backward compatibility)
            return True

        # Get API key from headers
        api_key = request.headers.get(self.header_name)

        if not api_key:
            logger.warning(f"Missing {self.header_name} header from {request.client.host}")
            return False

        if api_key != self.api_key:
            logger.warning(f"Invalid API key from {request.client.host}")
            return False

        return True

    def require_auth(self, request: Request):
        """Require authentication, raise exception if failed"""
        if not self.verify_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": f'ApiKey realm="ASR API"'}
            )

    async def __call__(self, request: Request, call_next):
        """FastAPI middleware call"""
        # Skip auth for health check endpoints
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Skip auth if no API key is configured
        if not self.api_key:
            return await call_next(request)

        # Verify API key
        self.require_auth(request)

        # Continue with request
        return await call_next(request)

# Global instance
_auth_instance = None

def get_auth() -> APIKeyAuth:
    """Get global authentication instance"""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = APIKeyAuth()
    return _auth_instance

def require_auth(request: Request):
    """Convenience function to require authentication"""
    get_auth().require_auth(request)