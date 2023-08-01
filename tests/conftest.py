import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base

TEST_DATABASE_URL = "postgresql://test_user:test_password@test_db:5432/test_db"


@pytest.fixture(scope="module")
def db_setup():
    engine = create_engine(TEST_DATABASE_URL)
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    Base.metadata.create_all(engine)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
    Base.metadata.drop_all(engine)
