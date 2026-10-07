from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongodb_uri: str = ""
    model_path: Path = Path("backend/saved_models")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_model_path(self) -> Path:
        path = self.model_path
        if not path.is_absolute():
            cwd = Path.cwd()
            if cwd.name == "backend" and path.parts and path.parts[0] == "backend":
                path = Path(*path.parts[1:])
            path = (cwd / path).resolve()
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
