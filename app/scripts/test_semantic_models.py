from io import BytesIO
from types import SimpleNamespace

import jwt
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import ingestion
from app.core import cube_client, db_manager, semantic_models
from app.ingestion.pipeline import IngestionService


def test_yaml_validation_rejects_bad_names_and_empty_files():
    with pytest.raises(ValueError, match="Nom de fichier"):
        semantic_models._validate_yaml_files({"../unsafe.yaml": "cube: Test"})
    with pytest.raises(ValueError, match="vide"):
        semantic_models._validate_yaml_files({"empty.yaml": "  "})
    with pytest.raises(ValueError, match="plusieurs fois"):
        semantic_models._validate_yaml_files(
            {"Cube.yaml": "cube: A", "cube.YAML": "cube: B"}
        )


def test_tenant_yaml_schemas_are_replaced_without_touching_other_tenants(
    monkeypatch, tmp_path
):
    schema_root = tmp_path / "cube"
    monkeypatch.setattr(
        semantic_models,
        "tenant_cube_schema_directory",
        lambda tenant: schema_root / tenant / "model" / "cubes",
    )
    previous = semantic_models.replace_tenant_cube_schemas(
        "isnov", {"invoice.yaml": "cubes:\n  - name: Invoice"}
    )
    assert previous == {}
    semantic_models.replace_tenant_cube_schemas(
        "another", {"sales.yml": "cubes:\n  - name: Sales"}
    )

    updated_previous = semantic_models.replace_tenant_cube_schemas(
        "isnov", {"customer.yaml": "cubes:\n  - name: Customer"}
    )

    assert updated_previous == {"invoice.yaml": "cubes:\n  - name: Invoice"}
    assert sorted(path.name for path in (schema_root / "isnov" / "model" / "cubes").iterdir()) == [
        "customer.yaml"
    ]
    assert (
        schema_root / "another" / "model" / "cubes" / "sales.yml"
    ).read_text(encoding="utf-8") == "cubes:\n  - name: Sales"


def test_lancedb_index_isolated_and_reindex_replaces_tenant_models(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        semantic_models,
        "get_cfg",
        lambda: SimpleNamespace(
            lancedb_uri=str(tmp_path / "lancedb"),
            ollama_base_url="http://ollama.test",
            embed_model="test-embedding",
        ),
    )
    monkeypatch.setattr(
        semantic_models, "_embed", lambda text: [float(len(text)), 1.0, 0.0]
    )

    assert semantic_models.index_tenant_cube_models(
        "isnov", {"invoice.yaml": "cubes:\n  - name: Invoice"}
    ) == 1
    semantic_models.index_tenant_cube_models(
        "another", {"sales.yaml": "cubes:\n  - name: Sales"}
    )
    semantic_models.index_tenant_cube_models(
        "isnov", {"customer.yaml": "cubes:\n  - name: Customer"}
    )

    assert semantic_models.get_tenant_cube_models("ISNOV") == [
        {"table_name": "customer", "yaml_content": "cubes:\n  - name: Customer"}
    ]
    assert semantic_models.get_tenant_cube_models("another") == [
        {"table_name": "sales", "yaml_content": "cubes:\n  - name: Sales"}
    ]


def test_pipeline_indexes_original_file_content_with_tenant_key(monkeypatch, tmp_path):
    source = "cubes:\n  - name: Raw YAML\n    sql: SELECT 1\n"
    lancedb_uri = str(tmp_path / "pipeline-lancedb")
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_cfg",
        lambda: SimpleNamespace(lancedb_uri=lancedb_uri, embed_retries=0),
    )
    monkeypatch.setattr(
        semantic_models,
        "get_cfg",
        lambda: SimpleNamespace(lancedb_uri=lancedb_uri),
    )
    embedded_inputs = []

    def embed(text):
        embedded_inputs.append(text)
        return [float(len(text)), 1.0, 0.0]

    service = IngestionService(
        registry=object(),
        store=object(),
        embed_fn=embed,
    )

    indexed = service.index_semantic_file(
        file_obj=BytesIO(source.encode("utf-8")),
        filename="raw_model.yaml",
        tenant_id="ISNOV",
    )

    assert indexed["tenant_id"] == "isnov"
    assert embedded_inputs == [source]
    assert semantic_models.get_tenant_cube_models("isnov") == [
        {"table_name": "raw_model", "yaml_content": source}
    ]


def test_empty_lancedb_table_lookup_returns_no_models(monkeypatch, tmp_path):
    monkeypatch.setattr(
        semantic_models,
        "get_cfg",
        lambda: SimpleNamespace(lancedb_uri=str(tmp_path / "empty")),
    )

    assert semantic_models.get_tenant_cube_models("isnov") == []


def test_cube_credentials_are_sent_in_a_signed_tenant_token(monkeypatch):
    monkeypatch.setattr(
        cube_client, "CUBE_API_SECRET", "test-secret-for-cube-jwt-validation-32"
    )
    headers = cube_client._cube_auth_headers(
        "ISNOV",
        {
            "type": "postgres",
            "host": "db.example",
            "port": 5432,
            "database": "analytics",
            "user": "cube_reader",
            "password": "db-password",
        },
    )

    token = headers["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(
        token, "test-secret-for-cube-jwt-validation-32", algorithms=["HS256"]
    )
    assert claims["tenantId"] == "isnov"
    assert claims["companyDbHost"] == "db.example"
    assert claims["companyDbUser"] == "cube_reader"
    assert claims["companyDbPassword"] == "db-password"


def test_connection_summary_does_not_expose_secret(monkeypatch):
    monkeypatch.setattr(
        db_manager,
        "_load",
        lambda: {
            "connection-id": {
                "tenant_id": "isnov",
                "name": "Analytics",
                "db_type": "postgres",
                "host": "db.example",
                "port": 5432,
                "database": "analytics",
                "username": "cube_reader",
                "password_enc": "encrypted-password",
            }
        },
    )

    summary = db_manager.get_latest_connection_summary("ISNOV")

    assert summary["username"] == "cube_reader"
    assert "password" not in summary
    assert "password_enc" not in summary


def test_semantic_model_api_indexes_for_authenticated_tenant(
    monkeypatch,
):
    monkeypatch.setattr(
        ingestion,
        "get_cfg",
        lambda: SimpleNamespace(admin_roles=("admin",)),
    )
    monkeypatch.setattr(
        db_manager,
        "get_latest_connection_config",
        lambda tenant: {
            "type": "postgres",
            "host": "db.example",
            "port": 5432,
            "database": "analytics",
            "user": "cube_reader",
            "password": "db-password",
        },
    )
    app = FastAPI()
    app.include_router(ingestion.router)
    app.dependency_overrides[ingestion.current_claims] = lambda: {
        "tenant_id": "ISNOV",
        "sub": "admin-1",
        "role": "admin",
    }
    app.dependency_overrides[ingestion.get_service] = lambda: SimpleNamespace(
        index_semantic_file=lambda **kwargs: {
            "tenant_id": kwargs["tenant_id"],
            "table_name": "sales",
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/ingest/semantic-models/index",
            files=[("files", ("sales.yaml", "cubes:\n  - name: Sales", "text/yaml"))],
        )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "isnov"
    assert response.json()["file_indexed"] == 1


def test_semantic_model_api_rejects_non_admin(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "get_cfg",
        lambda: SimpleNamespace(admin_roles=("admin",)),
    )
    app = FastAPI()
    app.include_router(ingestion.router)
    app.dependency_overrides[ingestion.current_claims] = lambda: {
        "tenant_id": "isnov",
        "sub": "client-1",
        "role": "user",
    }

    with TestClient(app) as client:
        response = client.post(
            "/ingest/semantic-models/index",
            files=[("files", ("sales.yaml", "cubes:\n  - name: Sales", "text/yaml"))],
        )

    assert response.status_code == 403


def test_semantic_model_api_indexes_upload_using_pipeline_service(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "get_cfg",
        lambda: SimpleNamespace(admin_roles=("admin",)),
    )
    monkeypatch.setattr(
        db_manager,
        "get_latest_connection_config",
        lambda tenant: {"host": "db.example"},
    )
    recorded = []

    def index_file(**kwargs):
        content = kwargs["file_obj"].read().decode("utf-8")
        recorded.append((kwargs["filename"], content, kwargs["tenant_id"]))
        return {
            "tenant_id": kwargs["tenant_id"],
            "table_name": "sales",
        }

    app = FastAPI()
    app.include_router(ingestion.router)
    app.dependency_overrides[ingestion.current_claims] = lambda: {
        "tenant_id": "ISNOV",
        "sub": "admin-1",
        "role": "admin",
    }
    app.dependency_overrides[ingestion.get_service] = lambda: SimpleNamespace(
        index_semantic_file=index_file
    )

    with TestClient(app) as client:
        response = client.post(
            "/ingest/semantic-models/index",
            files=[("files", ("sales.yaml", "cubes: []\n", "text/yaml"))],
        )

    assert response.status_code == 200
    assert recorded == [("sales.yaml", "cubes: []\n", "isnov")]


def test_semantic_model_api_rejects_duplicate_upload_filenames(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "get_cfg",
        lambda: SimpleNamespace(admin_roles=("admin",)),
    )
    app = FastAPI()
    app.include_router(ingestion.router)
    app.dependency_overrides[ingestion.current_claims] = lambda: {
        "tenant_id": "isnov",
        "sub": "admin-1",
        "role": "admin",
    }

    with TestClient(app) as client:
        response = client.post(
            "/ingest/semantic-models/index",
            files=[
                ("files", ("sales.yaml", "cubes: []", "text/yaml")),
                ("files", ("SALES.YAML", "cubes: []", "text/yaml")),
            ],
        )

    assert response.status_code == 400
