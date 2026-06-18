from sqlalchemy import Column, Integer, String, DateTime, func
from backend.database import Base

class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, nullable=False)
    total_rounds = Column(Integer, nullable=False)
    status = Column(String(16), default='pending', index=True)
    progress = Column(Integer, default=0)
    expected_clients = Column(Integer, nullable=False)
    received_clients = Column(Integer, default=0)
    model_version = Column(String(64))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    global_model_path = Column(String(256))
