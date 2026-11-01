"""Explicit server-side authorization for sector administration."""
import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


_bearer = HTTPBearer(auto_error=False)


def require_sector_admin(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    token = os.getenv("SECTOR_ADMIN_TOKEN", "")
    # Fail closed, including accidental weak configuration; no default admin.
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="Sector administration is disabled")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), token.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Valid administrator credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
