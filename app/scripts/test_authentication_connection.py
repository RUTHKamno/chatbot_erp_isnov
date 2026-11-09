from types import SimpleNamespace

from app import authentication


def test_local_auth_connection_maps_docker_postgres_hostname(monkeypatch):
    monkeypatch.delenv("AUTH_DATABASE_URL", raising=False)
    monkeypatch.delenv("AUTH_DATABASE_HOST", raising=False)
    monkeypatch.delenv("AUTH_DATABASE_PORT", raising=False)
    monkeypatch.setattr(
        authentication,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql+asyncpg://erp_user:secret@postgres:5432/erp_dwh"
        ),
    )

    url = authentication._authentication_database_url()

    assert url.host == "127.0.0.1"
    assert url.port == 5433
    assert url.database == "erp_dwh"
    assert url.username == "erp_user"


def test_local_auth_connection_uses_explicit_auth_database_url(monkeypatch):
    monkeypatch.setenv(
        "AUTH_DATABASE_URL",
        "postgresql://erp_user:secret@db.example.test:6543/erp_dwh",
    )

    url = authentication._authentication_database_url()

    assert url.host == "db.example.test"
    assert url.port == 6543


def test_local_auth_connection_supports_external_port_override(monkeypatch):
    monkeypatch.delenv("AUTH_DATABASE_URL", raising=False)
    monkeypatch.setenv("AUTH_DATABASE_HOST", "localhost")
    monkeypatch.setenv("AUTH_DATABASE_PORT", "5544")
    monkeypatch.setattr(
        authentication,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql://erp_user:secret@erp_chatbot_postgres:5432/erp_dwh"
        ),
    )

    url = authentication._authentication_database_url()

    assert url.host == "localhost"
    assert url.port == 5544


def test_auth_engine_uses_local_sync_driver_and_host(monkeypatch):
    monkeypatch.delenv("AUTH_DATABASE_URL", raising=False)
    monkeypatch.delenv("AUTH_DATABASE_HOST", raising=False)
    monkeypatch.delenv("AUTH_DATABASE_PORT", raising=False)
    monkeypatch.setattr(
        authentication,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql+asyncpg://erp_user:secret@postgres:5432/erp_dwh"
        ),
    )
    authentication.get_auth_engine.cache_clear()

    engine = authentication.get_auth_engine()
    try:
        assert engine.url.drivername == "postgresql+psycopg2"
        assert engine.url.host == "127.0.0.1"
        assert engine.url.port == 5433
    finally:
        engine.dispose()
        authentication.get_auth_engine.cache_clear()
