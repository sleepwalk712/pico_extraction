import datetime

from sqlalchemy import Column, Integer, ForeignKey, Text, DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base


class Result(Base):
    __tablename__ = 'results'
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('tasks.task_id'), nullable=False)
    data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    task = relationship('Task', back_populates='results')
