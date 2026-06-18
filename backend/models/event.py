from sqlalchemy import Column, Integer, String, Text, DateTime, func, JSON
from backend.database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), default='info')
    source = Column(String(64))
    message = Column(Text)
    metadata_info = Column("metadata", JSON)  # 'metadata' is reserved by SQLAlchemy, renaming attribute
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
