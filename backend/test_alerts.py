"""Alert credentials stay in headers; bad rules never reach provider evaluation."""
import json
from unittest.mock import patch

import pytest

import models
from realtime import evaluate_alerts
from main import limiter
from test_portfolio_access import client


@pytest.fixture(autouse=True)
def reset_request_limits():
    # Every case gets its own request budget, including explicit route limits.
    limiter.reset()


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
    with patch("realtime.evaluate_price_alert", return_value={"type": "price", "triggered": True, "ticker": "SBK.JO"}) as provider:
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
    price.assert_called_once_with("SBK.JO", "above", 100.0, was_triggered=False)


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


@pytest.mark.parametrize("condition,prices", [
    ("above", [120, 120, 120, 90, 110]),
    ("crosses_above", [120, 120, 120, 90, 110]),
    ("below", [80, 80, 80, 110, 90]),
    ("crosses_below", [80, 80, 80, 110, 90]),
])
def test_stored_price_alert_fires_once_and_rearms_on_opposite_side(client, condition, prices):
    api, db = client
    response = api.post("/api/alerts", headers=OWNER, json={**VALID, "condition": condition})
    assert response.status_code == 200
    fired = []
    with patch("realtime.get_live_quote", side_effect=[{"price": price} for price in prices]):
        for index in range(len(prices)):
            result = api.post("/api/alerts/check", headers=OWNER)
            assert result.status_code == 200
            fired.append(len(result.json()["fired"]))
            if index == 0:
                original_time = db.query(models.Alert).one().last_triggered_at
            if index in (1, 2, 3):
                assert db.query(models.Alert).one().last_triggered_at == original_time
            if index == 3:
                assert api.get("/api/alerts", headers=OWNER).json()["alerts"][0]["armed"]
    assert fired == [1, 0, 0, 0, 1]
    assert db.query(models.Alert).one().triggered


def test_quote_outage_does_not_rearm_a_latched_alert(client):
    api, db = client
    api.post("/api/alerts", headers=OWNER, json=VALID)
    with patch("realtime.get_live_quote", side_effect=[{"price": 120}, None, {"price": 120}]):
        assert len(api.post("/api/alerts/check", headers=OWNER).json()["fired"]) == 1
        assert api.post("/api/alerts/check", headers=OWNER).json()["results"] == []
        assert api.post("/api/alerts/check", headers=OWNER).json()["fired"] == []
    assert db.query(models.Alert).one().triggered


def test_stateless_price_alert_accepts_explicit_previous_state(client):
    api, _ = client
    with patch("realtime.get_live_quote", return_value={"price": 120}):
        response = api.post("/api/alerts/evaluate", json={"rules": [
            {"ticker": "SBK.JO", "threshold": 100, "condition": "crosses_above", "was_triggered": True},
        ]})
    assert response.status_code == 200
    assert response.json()["fired"] == []
    assert response.json()["results"][0]["condition_met"]
    assert not response.json()["results"][0]["armed"]


def test_stale_check_cannot_publish_an_edge_already_claimed_by_another_check(client):
    api, db = client
    alert_id = api.post("/api/alerts", headers=OWNER, json=VALID).json()["id"]

    def concurrent_evaluation(rules):
        assert not rules[0]["was_triggered"]
        # Simulate a competing check winning after this request read the old state.
        db.query(models.Alert).filter(models.Alert.id == alert_id).update(
            {models.Alert.triggered: True}, synchronize_session=False)
        evaluation = {"id": alert_id, "type": "price", "triggered": True, "condition_met": True, "message": "crossed"}
        return {"fired": [evaluation], "results": [evaluation]}

    with patch("main.evaluate_alerts", side_effect=concurrent_evaluation):
        result = api.post("/api/alerts/check", headers=OWNER)
    assert result.status_code == 200
    assert result.json()["fired"] == []
    assert not result.json()["results"][0]["triggered"]
