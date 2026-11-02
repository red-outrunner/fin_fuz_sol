"""Anonymous alert access uses a private, high-entropy bearer key."""
import re
import hashlib

from fastapi import Header, HTTPException


def require_alert_key(x_alert_key: str = Header(default="")) -> str:
    # Do not accept the legacy public client_key query parameter or shared default.
    if not re.fullmatch(r"[0-9a-f]{64}", x_alert_key):
        raise HTTPException(status_code=401, detail="A valid alert key is required")
    # Store only a fingerprint; a database read must not expose bearer credentials.
    return "alert:" + hashlib.sha256(x_alert_key.encode()).hexdigest()
