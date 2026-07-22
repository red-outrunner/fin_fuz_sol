from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    tier = Column(String, default="free") # free, pro, institutional
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False) # admin: full access to all tools


class Alert(Base):
    """Client-scoped price / volume / earnings alerts (no login required)."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    client_key = Column(String, index=True, nullable=False)
    ticker = Column(String, index=True, nullable=False)
    alert_type = Column(String, default="price")  # price | volume | earnings
    condition = Column(String, default="above")   # above | below
    threshold = Column(Float, nullable=True)
    label = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    triggered = Column(Boolean, default=False)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
