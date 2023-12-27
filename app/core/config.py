from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic import Field


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
    )

    POSTGRES_USER: str = Field(default=None)
    POSTGRES_PASSWORD: str = Field(default=None)
    POSTGRES_DB: str = Field(default=None)
    POSTGRES_HOST: str = Field(default=None)
    POSTGRES_PORT: int = Field(default=None)

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


class TestDatabaseConfig(DatabaseConfig):
    TEST_DATABASE_URL: str = "postgresql://test_user:test_password@test_db:5432/test_db"


Config = DatabaseConfig()
TestConfig = TestDatabaseConfig()
