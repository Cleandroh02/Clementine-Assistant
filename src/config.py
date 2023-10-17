from pydantic import BaseSettings


class Settings(BaseSettings):
    POSTGRESDB_URL: str = None
    POSTGRESDB_USER: str = None
    POSTGRESDB_PORT: int = None
    POSTGRESDB_PASSWORD: str = None
    POSTGRESDB_NAME: str = None

    class Config:
        env_file = None
        env_file_encoding = "utf-8"


def get_settings(env: str = None) -> BaseSettings:
    if env is not None:
        return Settings(_env_file=f"{env}.env")

    else:
        return Settings()
