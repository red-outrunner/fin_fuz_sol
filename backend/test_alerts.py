"""Alert credentials stay in headers; bad rules never reach provider evaluation."""
import json
from unittest.mock import patch

import pytest

import models
from realtime import evaluate_alerts
from test_portfolio_access import client


OWNER = {"X-Alert-Key": "a" * 64}
OTHER = {"X-Alert-Key": "b" * 64}
VALID = {"ticker": " sbk.jo ", "alert_type": "price", "condition": "above", "threshold": 100}


@pytest.mark.parametrize("key", [None, "", "default", "ck_old", "a" * 63, "z" * 64])
def test_stored_alert_routes_require_secure_header(client, key):
    api, db = client
    headers = {} if key is None else {"X-Alert-Key": key}
    assert api.get("/api/alerts", headers=headers).status_code == 401
    assert api.post("/api/alerts", json=VALID, headers=headers).status_code == 401
    assert api.delete("/api/alerts/1", headers=headers).status_code == 401
    assert api.post("/api/alerts/check", headers=headers).status_code == 401
    assert db.query(models.Alert).count() == 0


def test_private_alert_lifecycle_and_isolation(client):
    api, db = client
    response = api.post("/api/alerts", headers=OWNER, json=VALID)
    assert response.status_code == 200
    alert_id = response.json()["id"]
    row = db.query(models.Alert).one()
    assert row.client_key.startswith("alert:")
    assert OWNER["X-Alert-Key"] not in row.client_key
    assert row.ticker == "SBK.JO"
    assert api.get("/api/alerts", headers=OTHER).json() == {"alerts": []}
    assert len(api.get("/api/alerts", headers=OWNER).json()["alerts"]) == 1
    assert api.delete(f"/api/alerts/{alert_id}", headers=OTHER).status_code == 404
    with patch("realtime.evaluate_price_alert", return_value={"triggered": True, "ticker": "SBK.JO"}) as provider:
        assert api.post("/api/alerts/check", headers=OTHER).json()["results"] == []
        provider.assert_not_called()
        result = api.post("/api/alerts/check", headers=OWNER)
        assert result.status_code == 200
        assert result.json()["fired"][0]["id"] == alert_id
    assert api.delete(f"/api/alerts/{alert_id}", headers=OWNER).status_code == 200
    assert api.get("/api/alerts", headers=OWNER).json()["alerts"] == []


def test_legacy_keys_no_longer_authorize_access(client):
    api, _ = client
    assert api.get("/api/alerts", params={"client_key": "default"}).status_code == 401
    assert api.delete("/api/alerts/1", params={"client_key": "default"}).status_code == 401
    assert api.get("/api/alerts/ck_old").status_code in (404, 405)
    assert api.post("/api/alerts/ck_old/check").status_code == 404
    assert api.post("/api/alerts", headers=OWNER, json={**VALID, "client_key": "victim"}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("threshold", "bad"), ("threshold", None), ("threshold", -1),
    ("threshold", float("nan")), ("threshold", float("inf")),
    ("threshold", "Infinity"), ("condition", None), ("condition", "invalid"),
    ("ticker", " "), ("ticker", None), ("type", None), ("type", "invalid"),
])
def test_bad_rules_return_422_before_db_or_provider(client, field, value):
    api, db = client
    rule = {"ticker": "SBK.JO", "type": "price", "condition": "above", "threshold": 100, field: value}
    create = {("alert_type" if k == "type" else k): v for k, v in rule.items()}
    with patch("main.evaluate_alerts") as evaluate:
        for endpoint, payload in [("/api/alerts", create), ("/api/alerts/evaluate", {"rules": [rule]})]:
            response = api.post(endpoint, headers={**OWNER, "Content-Type": "application/json"},
                                content=json.dumps(payload))
            assert response.status_code == 422, response.text
        evaluate.assert_not_called()
    assert db.query(models.Alert).count() == 0


def test_earnings_alert_needs_no_threshold_and_volume_needs_one(client):
    api, _ = client
    assert api.post("/api/alerts", headers=OWNER,
                    json={"ticker": "SBK.JO", "alert_type": "earnings", "threshold": None}).status_code == 200
    assert api.post("/api/alerts", headers=OWNER,
                    json={"ticker": "SBK.JO", "alert_type": "volume"}).status_code == 422


def test_stateless_evaluation_remains_available(client):
    api, _ = client
    with patch("realtime.evaluate_price_alert", return_value={"triggered": True}) as price:
        response = api.post("/api/alerts/evaluate", json={"rules": [
            {"ticker": " sbk.jo ", "condition": "above", "threshold": 100},
        ]})
    assert response.status_code == 200
    price.assert_called_once_with("SBK.JO", "above", 100.0)


def test_corrupt_stored_rule_does_not_break_valid_alerts(client):
    api, db = client
    assert api.post("/api/alerts", headers=OWNER, json=VALID).status_code == 200
    owner_hash = db.query(models.Alert).one().client_key
    db.add(models.Alert(client_key=owner_hash, ticker="BAD.JO", condition="invalid", threshold=10))
    db.commit()
    with patch("realtime.evaluate_price_alert", return_value={"triggered": True}) as price:
        response = api.post("/api/alerts/check", headers=OWNER)
    assert response.status_code == 200
    assert len(response.json()["errors"]) == 1
    assert len(response.json()["fired"]) == 1
    price.assert_called_once()


def test_direct_evaluation_skips_malformed_rules():
    with patch("realtime.evaluate_price_alert") as price:
        result = evaluate_alerts([{"ticker": "SBK.JO", "threshold": "bad"},
                                  {"ticker": "SBK.JO", "condition": None, "threshold": 10}])
    price.assert_not_called()
    assert len(result["errors"]) == 2


def test_evaluation_batch_is_bounded(client):
    api, _ = client
    assert api.post("/api/alerts/evaluate", json={"rules": [
        {"ticker": "SBK.JO", "threshold": 100},
    ] * 51}).status_code == 422
