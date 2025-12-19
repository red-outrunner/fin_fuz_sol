from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    tier: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str
    tier: str # Return tier in token response for immediate frontend use

class TokenData(BaseModel):
    email: Optional[str] = None
