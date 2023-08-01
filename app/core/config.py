from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


class TestDatabaseConfig(DatabaseConfig):
    TEST_DATABASE_URL: str = "postgresql://test_user:test_password@test_db:5432/test_db"


Config = DatabaseConfig(_env_file=".env")
TestConfig = TestDatabaseConfig(_env_file=".env")
