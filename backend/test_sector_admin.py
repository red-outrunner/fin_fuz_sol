"""Administrative HTTP access must not trigger unauthenticated filesystem imports."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app, startup_event


TOKEN = "a" * 64
URL = "/api/admin/sync-sectors"


def test_startup_does_not_scan_server_directories():
    with patch("main.sync_sector_data") as sync:
        startup_event()
        sync.assert_called_once_with(auto_search_excel=False)


def test_unset_configuration_disables_admin(monkeypatch):
    monkeypatch.delenv("SECTOR_ADMIN_TOKEN", raising=False)
    with patch("main.sync_sector_data") as sync:
        assert TestClient(app).post(URL).status_code == 503
        sync.assert_not_called()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SECTOR_ADMIN_TOKEN", TOKEN)
    return TestClient(app)


@pytest.mark.parametrize("token", ["", "short"])
def test_disabled_without_strong_configuration(client, monkeypatch, token):
    monkeypatch.setenv("SECTOR_ADMIN_TOKEN", token)
    with patch("main.sync_sector_data") as sync:
        assert client.post(URL, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 503
        sync.assert_not_called()


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"},
                                      {"Authorization": f"Basic {TOKEN}"}])
def test_reject_unauthorized_callers(client, headers):
    with patch("main.sync_sector_data") as sync:
        assert client.post(URL, headers=headers).status_code == 401
        sync.assert_not_called()


def test_authorized_reload_never_searches_for_spreadsheets(client):
    with patch("main.sync_sector_data", return_value={"status": "success"}) as sync:
        response = client.post(URL, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        sync.assert_called_once_with(auto_search_excel=False)


def test_reject_server_path_even_for_admin(client):
    with patch("main.sync_sector_data") as sync, patch("sector_manager.import_from_excel") as importer:
        response = client.post(URL, params={"request_path": "/etc/private.xlsx"},
                               headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 400
        sync.assert_not_called()
        importer.assert_not_called()


def test_internal_error_does_not_expose_filesystem(client):
    with patch("main.sync_sector_data", side_effect=RuntimeError("/private/server/path")):
        response = client.post(URL, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 500
        assert "/private/server/path" not in response.text
