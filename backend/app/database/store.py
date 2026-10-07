import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.config import get_settings

logger = logging.getLogger(__name__)


class Store:
    """Mongo-backed history with a process-local adapter for zero-config development."""

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._db: Any = None
        self._iterations: list[dict[str, Any]] = []
        self._predictions: list[dict[str, Any]] = []
        uri = get_settings().mongodb_uri
        if uri:
            try:
                self._client = MongoClient(uri, serverSelectionTimeoutMS=2500)
                self._client.admin.command("ping")
                self._db = self._client["localized_weather"]
                logger.info("Connected to MongoDB Atlas")
            except Exception as exc:
                self._client = None
                self._db = None
                logger.warning("MongoDB connection failed (%s); using in-memory storage", exc)
        else:
            logger.warning("MONGODB_URI is unset; using in-memory storage for this process")

    def save_iteration(self, record: dict[str, Any]) -> None:
        if self._db is not None:
            self._db.iterations.insert_one(record.copy())
        else:
            self._iterations.append(record.copy())

    def history(self) -> list[dict[str, Any]]:
        records = list(self._db.iterations.find({}, {"_id": 0}).sort("iteration", 1)) if self._db is not None else self._iterations.copy()
        return records

    def latest(self) -> dict[str, Any] | None:
        if self._db is not None:
            return self._db.iterations.find_one({}, {"_id": 0}, sort=[("iteration", -1)])
        return self._iterations[-1].copy() if self._iterations else None

    def save_prediction(self, record: dict[str, Any]) -> None:
        record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if self._db is not None:
            self._db.predictions.insert_one(record.copy())
        else:
            self._predictions.append(record.copy())

    def predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._db is not None:
            return list(self._db.predictions.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
        return list(reversed(self._predictions[-limit:]))


store = Store()
