from sqlalchemy import Column, Integer, String, Boolean
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    tier = Column(String, default="free") # free, pro, institutional
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False) # admin: full access to all tools
