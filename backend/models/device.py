from sqlalchemy import Column, Integer, String, Float, DateTime, func, JSON
from backend.database import Base

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String(64), unique=True, nullable=False)
    name = Column(String(128))
    hardware_type = Column(String(64))
    status = Column(String(16), default='offline', index=True)
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    last_sync = Column(DateTime(timezone=True))
    contribution_score = Column(Float, default=0.0)
    region = Column(String(64))
    device_info = Column(JSON)
    contribution_weight = Column(Float, default=1.0)
    last_heartbeat = Column(DateTime(timezone=True))
    registered_at = Column(DateTime(timezone=True), server_default=func.now())
