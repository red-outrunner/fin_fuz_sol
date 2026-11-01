"""Exercise portfolio isolation against a disposable database, without providers."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import models
from database import Base, get_db
from main import app


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    def override_db():
        yield session
    app.dependency_overrides[get_db] = override_db
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
        engine.dispose()


def test_holding_can_only_be_deleted_by_its_portfolio(client):
    api, db = client
    owner = {"X-Portfolio-Key": "a" * 64}
    stranger = {"X-Portfolio-Key": "b" * 64}
    result = api.post("/api/portfolio/holdings", headers=owner,
                      json={"ticker": "SBK.JO", "quantity": 2, "avg_cost": 100})
    assert result.status_code == 200
    holding_id = result.json()["holding_id"]
    url = f"/api/portfolio/holdings/{holding_id}"
    assert api.delete(url).status_code == 401
    assert api.delete(url, params={"client_key": "default"}).status_code == 401
    assert api.delete(url, headers=stranger).status_code == 404
    assert db.query(models.PortfolioHolding).count() == 1
    assert api.get("/api/portfolio/holdings", headers=stranger).json()["count"] == 0
    assert api.delete(url, headers=owner).status_code == 200
    assert db.query(models.PortfolioHolding).count() == 0
    assert api.delete(url, headers=owner).status_code == 404


@pytest.mark.parametrize("key", [None, "", "default", "a" * 63, "g" * 64])
def test_all_portfolio_operations_require_private_key(client, key):
    api, _ = client
    headers = {} if key is None else {"X-Portfolio-Key": key}
    assert api.get("/api/portfolio/holdings", headers=headers).status_code == 401
    assert api.get("/api/portfolio/performance", headers=headers).status_code == 401
    assert api.post("/api/portfolio/holdings", headers=headers,
                    json={"ticker": "SBK.JO", "quantity": 1, "avg_cost": 100}).status_code == 401
