from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from backend.database import get_db
from backend.models import Event
from backend.schemas import ActivityItem, EventCreate
from backend.websocket.manager import manager
from datetime import datetime, timezone

router = APIRouter(prefix="/api/events", tags=["events"])

@router.post("", response_model=dict)
async def create_event(event: EventCreate, db: AsyncSession = Depends(get_db)):
    db_event = Event(
        event_type=event.event_type,
        severity=event.severity,
        source=event.source,
        message=event.message,
        metadata_info=event.metadata_info
    )
    db.add(db_event)
    await db.commit()
    await db.refresh(db_event)
    
    type_map = {
        "device.registered": "device_joined",
        "round.completed": "round_completed",
        "model.global_updated": "model_updated",
        "security.verified": "security_verified"
    }
    
    frontend_type = type_map.get(db_event.event_type, "device_joined")
    
    await manager.broadcast("events", {
        "event": "event.created",
        "data": ActivityItem(
            id=f"evt_{db_event.id}",
            type=frontend_type,
            message=db_event.message,
            timestamp=db_event.created_at.isoformat()
        ).model_dump()
    })
    
    return {"id": db_event.id}

@router.get("/activity", response_model=List[ActivityItem])
async def get_activity_feed(db: AsyncSession = Depends(get_db)):
    # Get latest events to populate the activity feed
    # Mapped to frontend expected types: "device_joined", "round_completed", "model_updated", "security_verified"
    result = await db.execute(
        select(Event)
        .where(Event.event_type.in_(["device.registered", "round.completed", "model.global_updated", "security.verified"]))
        .order_by(Event.created_at.desc())
        .limit(10)
    )
    events = result.scalars().all()
    
    # Map internal event types to frontend types
    type_map = {
        "device.registered": "device_joined",
        "round.completed": "round_completed",
        "model.global_updated": "model_updated",
        "security.verified": "security_verified"
    }
    
    if not events:
        # Return some initial dummy data if no events exist yet
        return [
            ActivityItem(
                id="evt_init_1",
                type="device_joined",
                message="Node-Alpha initialized",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        ]
        
    return [
        ActivityItem(
            id=f"evt_{e.id}",
            type=type_map.get(e.event_type, "device_joined"),
            message=e.message,
            timestamp=e.created_at.isoformat()
        ) for e in events
    ]
