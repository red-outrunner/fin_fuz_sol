from sqlalchemy import Integer, String, Boolean, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional
from database import Base


class DailyStockPick(Base):
    """The actual first published selection for a Johannesburg calendar day."""
    __tablename__ = "daily_stock_picks"

    date: Mapped[str] = mapped_column(String(10), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tier: Mapped[Optional[str]] = mapped_column(String, default="free", nullable=True) # free, pro, institutional
    is_active: Mapped[Optional[bool]] = mapped_column(Boolean, default=True, nullable=True)
    is_admin: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True) # admin: full access to all tools


class Alert(Base):
    """Client-scoped price / volume / earnings alerts (no login required)."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    client_key: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String, index=True, nullable=False)
    alert_type: Mapped[Optional[str]] = mapped_column(String, default="price", nullable=True)  # price | volume | earnings
    condition: Mapped[Optional[str]] = mapped_column(String, default="above", nullable=True)   # above | below
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    active: Mapped[Optional[bool]] = mapped_column(Boolean, default=True, nullable=True)
    triggered: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)


class PortfolioHolding(Base):
    """Client-scoped portfolio holdings (no login required)."""
    __tablename__ = "portfolio_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    client_key: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String, index=True, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # Average cost per share
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True)
