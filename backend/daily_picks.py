"""Persist published daily picks so yesterday never depends on today's provider data."""
from sqlalchemy.exc import IntegrityError

from database import SessionLocal
from models import DailyStockPick


def get_published_pick(day):
    with SessionLocal() as db:
        row = db.get(DailyStockPick, day)
        return {"ticker": row.ticker, "name": row.name} if row else None


def publish_pick(day, ticker, name):
    with SessionLocal() as db:
        db.add(DailyStockPick(date=day, ticker=ticker, name=name))
        try:
            db.commit()
        except IntegrityError:
            # Another worker published first. Its selection is authoritative.
            db.rollback()
        row = db.get(DailyStockPick, day)
        return {"ticker": row.ticker, "name": row.name}
