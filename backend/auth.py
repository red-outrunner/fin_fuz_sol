from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
import models
import os
import secrets

# Secret key from environment variable (REQUIRED in production).
# We never ship a hardcoded secret: a committed default would let anyone reading
# the repo forge valid JWTs. In production (ENVIRONMENT=production) a missing
# SECRET_KEY is a hard failure. In development we generate a random ephemeral key
# so there is no secret in source control (tokens just won't survive a restart).
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    if ENVIRONMENT == "production":
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. Refusing to start in "
            "production without a secret key (JWTs would be forgeable). Set "
            "SECRET_KEY to a strong random value (e.g. `openssl rand -hex 32`)."
        )
    SECRET_KEY = secrets.token_hex(32)
    print(
        "⚠️  WARNING: SECRET_KEY not set; generated an ephemeral development key. "
        "Tokens will be invalidated on restart. Set SECRET_KEY for stable sessions."
    )
else:
    print("✅ Using SECRET_KEY from environment variable")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
