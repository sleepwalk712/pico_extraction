import datetime

from sqlalchemy import Column, Integer, String, DateTime

from app.db.base import Base


class ModelVersion(Base):
    __tablename__ = 'model_versions'
    version_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    version_number = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
