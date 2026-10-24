"""Exact browser-origin configuration shared by the CLI and HTTP runtime."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from ipaddress import IPv6Address
from typing import cast
from urllib.parse import urlsplit

CORS_ALLOWED_ORIGINS_ENV = "AGENTBRICKS_CORS_ALLOWED_ORIGINS"


def validate_allowed_origins(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("CORS allowed_origins must be an array of HTTP(S) origins.")
    origins: list[str] = []
    for origin in value:
        try:
            if (
                not isinstance(origin, str)
                or not origin.isascii()
                or any(character.isspace() or ord(character) < 32 for character in origin)
                or any(character in origin for character in "*\\%?#")
            ):
                raise ValueError
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.path
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError
            host = parsed.hostname.lower()
            if ":" in host:
                address = IPv6Address(host)
                host = address.compressed
                if address.ipv4_mapped is not None:
                    # Browsers serialize mapped IPv4 addresses as hexadecimal IPv6 pieces.
                    mapped = int(address.ipv4_mapped)
                    host = f"::ffff:{mapped >> 16:x}:{mapped & 0xFFFF:x}"
                host = f"[{host}]"
            port = parsed.port
            default_port = 80 if parsed.scheme == "http" else 443
            authority = f"{host}:{port}" if port is not None and port != default_port else host
            origins.append(f"{parsed.scheme}://{authority}")
        except ValueError as exc:
            raise ValueError(
                "Invalid CORS origin. Use an exact HTTP(S) origin such as "
                "https://chat.example.com or http://localhost:3000, without a path or wildcard. "
                "Use ASCII/punycode hostnames for internationalized domains."
            ) from exc
    return tuple(dict.fromkeys(origins))


def allowed_origins_from_manifest(document: Mapping[str, object]) -> tuple[str, ...]:
    if "cors" not in document:
        return ()
    cors = document["cors"]
    if not isinstance(cors, Mapping):
        raise ValueError("agent.toml [cors] must be a table.")
    unexpected = cors.keys() - {"allowed_origins"}
    if unexpected:
        raise ValueError(f"Unsupported CORS settings: {', '.join(sorted(unexpected))}.")
    settings = cast(Mapping[str, object], cors)
    return validate_allowed_origins(settings.get("allowed_origins", []))


def allowed_origins_from_environment() -> tuple[str, ...]:
    raw = os.getenv(CORS_ALLOWED_ORIGINS_ENV)
    if raw is None:
        return ()
    try:
        return validate_allowed_origins(json.loads(raw))
    except ValueError as exc:
        raise ValueError(f"Invalid {CORS_ALLOWED_ORIGINS_ENV}: {exc}") from exc
