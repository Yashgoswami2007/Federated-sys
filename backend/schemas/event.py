from pydantic import BaseModel
from typing import Optional, Dict, Any

class ActivityItem(BaseModel):
    id: str
    type: str
    message: str
    timestamp: str

    class Config:
        from_attributes = True

class EventCreate(BaseModel):
    event_type: str
    severity: str = "info"
    source: Optional[str] = None
    message: str
    metadata_info: Optional[Dict[str, Any]] = None
