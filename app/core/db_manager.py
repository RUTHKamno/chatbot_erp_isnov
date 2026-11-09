"""Connexions multi-bases de données (PostgreSQL, MariaDB, MySQL, etc.) + lecture des lignes à indexer."""

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Tuple

from cryptography.fernet import Fernet
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import URL, Engine

from app.ingestion.settings import get_cfg

MAX_ROWS = 20000
_SYSTEM_SCHEMAS = {
    "information_schema",
    "pg_catalog",
    "pg_toast",
    "performance_schema",
    "sys",
    "mysql",
}
_lock = threading.Lock()
_engines: Dict[str, Engine] = {}


# ---------------------------------------------------------------- stockage des connexions
def _fernet() -> Fernet:
    key = os.environ.get("DB_CRED_KEY")
    if not key:
        raise ValueError("Variable d'environnement DB_CRED_KEY absente.")
    return Fernet(key.encode())


def _store_path():
    return get_cfg().data_dir / "db_connections.json"


def _load() -> Dict[str, Any]:
    p = _store_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(data: Dict[str, Any]) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean(v: Any) -> str:
    return " ".join(str(v).split()) if v is not None else ""


# ---------------------------------------------------------------- fabrique de connexions (Multi-DB)
def _make_engine(
    db_type: str, host: str, port: int, database: str, username: str, password: str
) -> Engine:
    """
    Sépare la configuration de la connexion selon le type de base de données (Postgres, MariaDB, MySQL, etc.).
    Utilise SQLAlchemy pour s'adapter au dialecte cible.
    """
    db_type = (db_type or "postgres").lower().strip()

    if db_type in ("postgres", "postgresql"):
        driver = "postgresql+psycopg2"
        port = port or 5432
        connect_args = {
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c statement_timeout=30000",
        }
    elif db_type in ("mariadb", "mysql"):
        driver = "mysql+pymysql"
        port = port or 3306
        connect_args = {
            "connect_timeout": 5,
        }
    else:
        raise ValueError(f"Type de base de données non supporté : {db_type}")

    url = URL.create(
        driver,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def test_connection(
    db_type: str, host: str, port: int, database: str, username: str, password: str
) -> None:
    engine = _make_engine(db_type, host, port, database, username, password)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as e:
        raise ValueError(f"Connexion impossible : {str(e).splitlines()[0][:200]}")
    finally:
        engine.dispose()


def save_connection(
    tenant_id: str,
    name: str,
    db_type: str,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
    show_all_schemas: bool,
) -> str:
    test_connection(
        db_type, host, port, database, username, password
    )  # Vérifie que la connexion est valide
    connection_id = uuid.uuid4().hex
    with _lock:
        data = _load()
        data[connection_id] = {
            "tenant_id": tenant_id,
            "name": name,
            "db_type": db_type,
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password_enc": _fernet().encrypt(password.encode()).decode(),
            "show_all_schemas": show_all_schemas,
        }
        _save(data)
    return connection_id


def list_connections(tenant_id: str) -> List[Dict[str, Any]]:
    keys = (
        "name",
        "db_type",
        "host",
        "port",
        "database",
        "username",
        "show_all_schemas",
    )
    return [
        {
            "connection_id": cid,
            **{k: c.get(k, "postgres") if k == "db_type" else c[k] for k in keys},
        }
        for cid, c in _load().items()
        if c["tenant_id"] == tenant_id
    ]


def get_latest_connection_config(tenant_id: str) -> Dict[str, Any] | None:
    """Return the latest saved tenant connection with its decrypted credentials."""
    from app.ingestion.pipeline import normalize_tenant_id

    normalized_tenant_id = normalize_tenant_id(tenant_id)
    for connection in reversed(list(_load().values())):
        if connection["tenant_id"] not in {tenant_id, normalized_tenant_id}:
            continue
        password = _fernet().decrypt(connection["password_enc"].encode()).decode()
        return {
            "type": connection.get("db_type", "postgres"),
            "host": connection["host"],
            "port": connection["port"],
            "database": connection["database"],
            "user": connection["username"],
            "password": password,
        }
    return None


def get_latest_connection_summary(tenant_id: str) -> Dict[str, Any] | None:
    """Return display-safe metadata for the latest connection of a tenant."""
    from app.ingestion.pipeline import normalize_tenant_id

    normalized_tenant_id = normalize_tenant_id(tenant_id)
    for connection_id, connection in reversed(list(_load().items())):
        if connection["tenant_id"] not in {tenant_id, normalized_tenant_id}:
            continue
        return {
            "connection_id": connection_id,
            "name": connection["name"],
            "db_type": connection.get("db_type", "postgres"),
            "host": connection["host"],
            "port": connection["port"],
            "database": connection["database"],
            "username": connection["username"],
        }
    return None


def _get_engine(connection_id: str, tenant_id: str) -> Tuple[Engine, Dict[str, Any]]:
    c = _load().get(connection_id)
    if not c or c["tenant_id"] != tenant_id:  # Isolation stricte entre tenants
        raise ValueError("Connexion introuvable.")
    with _lock:
        if connection_id not in _engines:
            password = _fernet().decrypt(c["password_enc"].encode()).decode()
            db_type = c.get("db_type", "postgres")
            _engines[connection_id] = _make_engine(
                db_type, c["host"], c["port"], c["database"], c["username"], password
            )
    return _engines[connection_id], c


# ---------------------------------------------------------------- exploration (Géré par SQLAlchemy de façon agnostique)
def list_schemas(connection_id: str, tenant_id: str) -> List[str]:
    engine, c = _get_engine(connection_id, tenant_id)
    insp = inspect(engine)
    names = [
        s
        for s in insp.get_schema_names()
        if s not in _SYSTEM_SCHEMAS and not s.startswith("pg_")
    ]
    return (
        sorted(names)
        if c["show_all_schemas"]
        else [s for s in names if s in ("public", c["database"])]
    )


def list_tables(
    connection_id: str, tenant_id: str, schema: str
) -> List[Dict[str, str]]:
    if schema not in list_schemas(connection_id, tenant_id):
        raise ValueError(f"Schéma inconnu ou non autorisé : {schema}")
    engine, c = _get_engine(connection_id, tenant_id)
    insp = inspect(engine)
    # Pour MySQL/MariaDB, le schéma correspond souvent à la base de données (schema=None ou schema=database)
    target_schema = None if c.get("db_type") in ("mysql", "mariadb") else schema

    tables = [
        {"name": n, "type": "table"}
        for n in sorted(insp.get_table_names(schema=target_schema))
    ]
    views = [
        {"name": n, "type": "view"}
        for n in sorted(insp.get_view_names(schema=target_schema))
    ]
    return tables + views


def _reflect(connection_id: str, tenant_id: str, schema: str, table: str) -> Table:
    """Valide schéma/table contre le catalogue réel."""
    if table not in [t["name"] for t in list_tables(connection_id, tenant_id, schema)]:
        raise ValueError(f"Table inconnue : {schema}.{table}")
    engine, c = _get_engine(connection_id, tenant_id)
    target_schema = None if c.get("db_type") in ("mysql", "mariadb") else schema
    return Table(table, MetaData(), schema=target_schema, autoload_with=engine)


def list_columns(
    connection_id: str, tenant_id: str, schema: str, table: str
) -> Dict[str, Any]:
    t = _reflect(connection_id, tenant_id, schema, table)
    engine = _get_engine(connection_id, tenant_id)[0]
    with engine.connect() as conn:
        rows = conn.execute(select(t).limit(3)).mappings().all()
    return {
        "columns": [
            {"name": c.name, "type": str(c.type), "primary_key": c.primary_key}
            for c in t.columns
        ],
        "sample": [{k: _clean(v)[:120] for k, v in r.items()} for r in rows],
    }


# ---------------------------------------------------------------- lecture selon la correspondance
def _col(t: Table, name: str):
    if name not in t.c.keys():
        raise ValueError(f"Colonne inconnue : {t.name}.{name}")
    return t.c[name]


def _resolve(tables: Dict[str, Table], ref: str):
    table, _, column = ref.partition(".")
    if table not in tables:
        raise ValueError(f"Table non sélectionnée : {table}")
    return _col(tables[table], column)


def _validate_mapping(m: Any) -> None:
    if not isinstance(m, dict) or not isinstance(m.get("base"), dict):
        raise ValueError("mapping invalide : « base » (schema, table) est obligatoire.")
    body = m.get("body")
    if not body or not all(
        isinstance(b, dict) and b.get("column") and b.get("label") for b in body
    ):
        raise ValueError(
            "Choisissez au moins une colonne à indexer, chacune avec un libellé."
        )
    if not m.get("roles"):
        raise ValueError(
            "Indiquez d'où viennent les rôles autorisés (colonne ou valeur fixe)."
        )


def _mapping_refs(m: Dict[str, Any]) -> List[str]:
    refs = [b["column"] for b in m["body"]]
    for key in ("module", "ecran", "roles", "url"):
        col = (m.get(key) or {}).get("column")
        if col:
            refs.append(col)
    return list(dict.fromkeys(refs))


def read_mapped_rows(
    connection_id: str, tenant_id: str, mapping: Dict[str, Any]
) -> List[Dict[str, Any]]:
    _validate_mapping(mapping)
    engine = _get_engine(connection_id, tenant_id)[0]
    base = mapping["base"]
    tables = {
        base["table"]: _reflect(connection_id, tenant_id, base["schema"], base["table"])
    }
    source = tables[base["table"]]

    for j in mapping.get("joins", []):  # Jointures LEFT JOIN, chaînables
        if j["table"] in tables:
            raise ValueError("Une même table ne peut être reliée qu'une seule fois.")
        right = _reflect(connection_id, tenant_id, j["schema"], j["table"])
        left_col = _resolve(tables, f"{j['left_table']}.{j['left_column']}")
        source = source.join(
            right, left_col == _col(right, j["right_column"]), isouter=True
        )
        tables[j["table"]] = right

    refs = _mapping_refs(mapping)
    cols = [_resolve(tables, r).label(f"c{i}") for i, r in enumerate(refs)]
    stmt = select(*cols).select_from(source).limit(MAX_ROWS)
    with engine.connect() as conn:
        raw = conn.execute(stmt).mappings().all()
    return [{r: row[f"c{i}"] for i, r in enumerate(refs)} for row in raw]


def _pick(spec: Any, row: Dict[str, Any], default: str = "") -> str:
    spec = spec or {}
    if spec.get("column"):
        return _clean(row.get(spec["column"])) or default
    return _clean(spec.get("value")) or default


def build_blocks(
    rows: List[Dict[str, Any]], mapping: Dict[str, Any], tenant_id: str
) -> Tuple[List[str], int]:
    """Un bloc par ligne, au format d'en-tête attendu par ingest_formatted_texts."""
    blocks: List[str] = []
    skipped = 0
    for row in rows:
        parts = []
        for b in mapping["body"]:
            v = _clean(row.get(b["column"]))
            if v:
                parts.append(f"{b['label']} : {v}" + ("" if v[-1] in ".!?" else "."))
        roles = _pick(mapping.get("roles"), row)
        if not parts or not roles:  # Sans rôle : on n'indexe pas
            skipped += 1
            continue
        header = (
            f"Tenant : {tenant_id}\n"
            f"Module : {_pick(mapping.get('module'), row, '-')}\n"
            f"Rôles autorisés : {roles}\n"
            f"Écran : {_pick(mapping.get('ecran'), row, '-')}\n"
            f"URL : {_pick(mapping.get('url'), row)}\n"
            "---\n"
        )
        blocks.append(header + " ".join(parts))
    return blocks, skipped
