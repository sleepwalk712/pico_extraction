from sqlalchemy import text
from app.db.session import SessionLocal


def test_connection():
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT 1")).scalar_one()
        assert result == 1
    finally:
        session.close()
