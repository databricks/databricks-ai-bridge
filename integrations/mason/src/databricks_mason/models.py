"""Typed, forward-compatible views over agents/v1 JSON responses.

Every model subclasses `dict` and holds the complete raw response, so mapping
access (`store["name"]`), `json.dumps`, and any new/unknown server fields keep
working. Subclasses add typed `@property` accessors as a convenience layer;
they never mutate the stored data, so nothing is lost as the preview API grows.

Accessors are intentionally named to avoid shadowing `dict` methods (e.g. list
models expose `.stores` / `.session_items`, never `.items`).
"""

from __future__ import annotations

from typing import Any, Optional


class _Model(dict):
    """A dict-backed API response with typed accessors over the raw JSON."""

    def to_dict(self) -> dict:
        """A plain-dict shallow copy of the raw response."""
        return dict(self)


def _wrap(value: Any, cls: type) -> Any:
    return cls(value) if isinstance(value, dict) else None


def _wrap_list(value: Any, cls: type) -> list:
    return [cls(v) for v in value if isinstance(v, dict)] if isinstance(value, list) else []


class StorageBackend(_Model):
    @property
    def backend_type(self) -> Optional[str]:
        return self.get("backend_type")

    @property
    def backend_id(self) -> Optional[str]:
        return self.get("backend_id")


class MemoryStore(_Model):
    @property
    def name(self) -> Optional[str]:
        return self.get("name")

    @property
    def display_name(self) -> Optional[str]:
        return self.get("display_name")

    @property
    def workspace_id(self) -> Optional[str]:
        return self.get("workspace_id")

    @property
    def owner_user_id(self) -> Optional[str]:
        return self.get("owner_user_id")

    @property
    def description(self) -> Optional[str]:
        return self.get("description")

    @property
    def storage_backend(self) -> Optional[StorageBackend]:
        return _wrap(self.get("storage_backend"), StorageBackend)

    @property
    def create_time(self) -> Optional[str]:
        return self.get("create_time")

    @property
    def update_time(self) -> Optional[str]:
        return self.get("update_time")


class MemoryEntry(_Model):
    @property
    def name(self) -> Optional[str]:
        return self.get("name")

    @property
    def actor_id(self) -> Optional[str]:
        return self.get("actor_id")

    @property
    def path(self) -> Optional[str]:
        return self.get("path")

    @property
    def content(self) -> Optional[str]:
        return self.get("content")

    @property
    def description(self) -> Optional[str]:
        return self.get("description")

    @property
    def session_id(self) -> Optional[str]:
        return self.get("session_id")

    @property
    def source_type(self) -> Optional[str]:
        return self.get("source_type")

    @property
    def create_time(self) -> Optional[str]:
        return self.get("create_time")

    @property
    def update_time(self) -> Optional[str]:
        return self.get("update_time")


class MemoryStoreList(_Model):
    @property
    def stores(self) -> list[MemoryStore]:
        return _wrap_list(self.get("managed_memory_stores"), MemoryStore)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class MemoryEntryList(_Model):
    @property
    def entries(self) -> list[MemoryEntry]:
        return _wrap_list(self.get("managed_memory_entries"), MemoryEntry)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class MemorySearchHit(_Model):
    @property
    def entry(self) -> Optional[MemoryEntry]:
        return _wrap(self.get("managed_memory_entry"), MemoryEntry)

    @property
    def score(self) -> Optional[float]:
        return self.get("score")


class MemorySearchResult(_Model):
    @property
    def entries(self) -> list[MemoryEntry]:
        return _wrap_list(self.get("managed_memory_entries"), MemoryEntry)

    @property
    def results(self) -> list[MemorySearchHit]:
        return _wrap_list(self.get("results"), MemorySearchHit)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class SessionStore(_Model):
    @property
    def session_store_name(self) -> Optional[str]:
        return self.get("session_store_name")

    @property
    def session_store_id(self) -> Optional[str]:
        return self.get("session_store_id")

    @property
    def creator_user_id(self) -> Optional[str]:
        return self.get("creator_user_id")

    @property
    def description(self) -> Optional[str]:
        return self.get("description")

    @property
    def create_time(self) -> Optional[str]:
        return self.get("create_time")

    @property
    def update_time(self) -> Optional[str]:
        return self.get("update_time")


class SessionStoreList(_Model):
    @property
    def stores(self) -> list[SessionStore]:
        return _wrap_list(self.get("session_stores"), SessionStore)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class Session(_Model):
    @property
    def session_store_name(self) -> Optional[str]:
        return self.get("session_store_name")

    @property
    def session_id(self) -> Optional[str]:
        return self.get("session_id")

    @property
    def actor_id(self) -> Optional[str]:
        return self.get("actor_id")

    @property
    def root_session_id(self) -> Optional[str]:
        return self.get("root_session_id")

    @property
    def parent_session_id(self) -> Optional[str]:
        return self.get("parent_session_id")

    @property
    def metadata(self) -> Optional[dict]:
        return self.get("metadata")

    @property
    def create_time(self) -> Optional[str]:
        return self.get("create_time")

    @property
    def update_time(self) -> Optional[str]:
        return self.get("update_time")

    @property
    def last_activity_time(self) -> Optional[str]:
        return self.get("last_activity_time")


class SessionList(_Model):
    @property
    def sessions(self) -> list[Session]:
        return _wrap_list(self.get("sessions"), Session)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class SessionItem(_Model):
    @property
    def item_id(self) -> Optional[str]:
        return self.get("item_id")

    @property
    def data(self) -> Any:
        return self.get("data")

    @property
    def create_time(self) -> Optional[str]:
        return self.get("create_time")


class SessionItemList(_Model):
    @property
    def session_items(self) -> list[SessionItem]:
        return _wrap_list(self.get("session_items"), SessionItem)

    @property
    def next_page_token(self) -> Optional[str]:
        return self.get("next_page_token")


class PoppedSessionItem(_Model):
    @property
    def item(self) -> Optional[SessionItem]:
        return _wrap(self.get("item"), SessionItem)
