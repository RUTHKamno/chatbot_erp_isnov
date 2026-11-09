"""Authentication and tenant-user management for the Streamlit application."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import jwt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.engine.url import URL

from app.config import get_settings

_PASSWORD_SCHEME = "scrypt"
_PASSWORD_N = 2**14
_PASSWORD_R = 8
_PASSWORD_P = 1


@lru_cache(maxsize=1)
def get_auth_engine() -> Engine:
    url = _authentication_database_url()
    print("authenticate url", url)
    if url.get_backend_name() == "postgresql":
        url = url.set(drivername="postgresql+psycopg2")
        print("url", url)
    return create_engine(url, pool_pre_ping=True)


def _authentication_database_url() -> URL:
    auth_database_url = os.getenv("AUTH_DATABASE_URL")
    url = make_url(auth_database_url or get_settings().database_url)

    docker_hosts = {"postgres", "erp_chatbot_postgres"}
    if (
        not Path("/.dockerenv").exists()
        and url.host in docker_hosts
        and not auth_database_url
    ):
        url = url.set(
            host=os.getenv("AUTH_DATABASE_HOST", "127.0.0.1"),
            port=int(os.getenv("AUTH_DATABASE_PORT", "5433")),
        )
    return url


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_N,
        r=_PASSWORD_R,
        p=_PASSWORD_P,
        dklen=64,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{_PASSWORD_SCHEME}${_PASSWORD_N}${_PASSWORD_R}${_PASSWORD_P}${salt_text}${digest_text}"


def verify_password(password: str, stored_password: str) -> tuple[bool, bool]:
    """Return (valid, needs_rehash); legacy plaintext is migrated on successful login."""
    if not stored_password.startswith(f"{_PASSWORD_SCHEME}$"):
        return hmac.compare_digest(password, stored_password), True

    try:
        scheme, n, r, p, salt_text, digest_text = stored_password.split("$", 5)
        if scheme != _PASSWORD_SCHEME:
            return False, False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False, False
    return hmac.compare_digest(actual, expected), False


def _public_user(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "surname": row["surname"],
        "role": (row["role"] or "user").strip().lower(),
        "tenant_id": row["tenant_id"],
    }


def authenticate_user(
    tenant_id: str, name: str, surname: str, password: str
) -> dict[str, Any] | None:
    statement = text("""
        SELECT id, name, surname, password, role, tenant_id
        FROM users.users_tests
        WHERE lower(name) = lower(:name)
          AND lower(surname) = lower(:surname)
          AND lower(tenant_id) = lower(:tenant_id)
        """)
    print("after the query select", statement)
    with get_auth_engine().begin() as connection:
        print("enter the first condition")
        candidates = connection.execute(
            statement,
            {
                "name": name.strip(),
                "surname": surname.strip(),
                "tenant_id": tenant_id.strip(),
            },
        ).mappings()
        print("candidates", candidates)
        for row in candidates:
            valid, needs_rehash = verify_password(password, row["password"])
            if not valid:
                print("not valid", valid)
                continue
            if needs_rehash:
                print("needs_rehash", needs_rehash)
                connection.execute(
                    text(
                        "UPDATE users.users_tests SET password = :password WHERE id = :id"
                    ),
                    {"password": hash_password(password), "id": row["id"]},
                )
            return _public_user(row)
    return None


def create_tenant_user(
    tenant_id: str, name: str, surname: str, password: str, role: str
) -> dict[str, Any]:
    tenant = tenant_id.strip()
    first_name = name.strip()
    last_name = surname.strip()
    normalized_role = role.strip().lower()
    allowed_roles = {"admin", "bi_analyst", "user"}
    if not tenant or len(tenant) > 100:
        raise ValueError(
            "L’identifiant de l’entreprise doit contenir 1 à 100 caractères."
        )
    if not first_name or len(first_name) > 100:
        raise ValueError("Le prénom doit contenir 1 à 100 caractères.")
    if not last_name or len(last_name) > 100:
        raise ValueError("Le nom doit contenir 1 à 100 caractères.")
    if not password:
        raise ValueError("Le mot de passe est obligatoire.")
    if normalized_role not in allowed_roles:
        raise ValueError("Le rôle sélectionné n’est pas autorisé.")

    with get_auth_engine().begin() as connection:
        duplicate = connection.execute(
            text("""
                SELECT 1
                FROM users.users_tests
                WHERE lower(name) = lower(:name)
                  AND lower(surname) = lower(:surname)
                  AND lower(tenant_id) = lower(:tenant_id)
                LIMIT 1
                """),
            {"name": first_name, "surname": last_name, "tenant_id": tenant},
        ).first()
        if duplicate:
            raise ValueError(
                "Un utilisateur portant ce nom existe déjà dans l’entreprise."
            )

        row = (
            connection.execute(
                text("""
                INSERT INTO users.users_tests (name, surname, password, role, tenant_id)
                VALUES (:name, :surname, :password, :role, :tenant_id)
                RETURNING id, name, surname, role, tenant_id
                """),
                {
                    "name": first_name,
                    "surname": last_name,
                    "password": hash_password(password),
                    "role": normalized_role,
                    "tenant_id": tenant,
                },
            )
            .mappings()
            .one()
        )
    return _public_user(row)


def list_tenant_users(tenant_id: str) -> list[dict[str, Any]]:
    with get_auth_engine().connect() as connection:
        rows = connection.execute(
            text("""
                SELECT id, name, surname, role, tenant_id
                FROM users.users_tests
                WHERE lower(tenant_id) = lower(:tenant_id)
                ORDER BY surname, name, id
                """),
            {"tenant_id": tenant_id.strip()},
        ).mappings()
        return [_public_user(row) for row in rows]


def create_access_token(user: dict[str, Any]) -> str:
    settings = get_settings()
    if not settings.jwt_secret_key or settings.jwt_secret_key == "change-me":
        raise RuntimeError("JWT_SECRET_KEY doit être configurée avec une clé forte.")
    payload = {
        "sub": str(user["id"]),
        "tenant_id": user["tenant_id"],
        "role": user["role"],
        "name": user["name"],
        "exp": datetime.now(timezone.utc)
        + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )
