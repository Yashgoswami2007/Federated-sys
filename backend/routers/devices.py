from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Dict

from backend.database import get_db
from backend.models import Device
from backend.schemas import EdgeDevice, RegionData, DeviceRegister, DeviceHeartbeat
from backend.websocket.manager import manager
from datetime import datetime, timezone

router = APIRouter(prefix="/api/devices", tags=["devices"])

@router.post("/register", response_model=EdgeDevice)
async def register_device(device_reg: DeviceRegister, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Device).where(Device.client_id == device_reg.client_id))
    device = result.scalars().first()

    if device:
        device.hardware_type = device_reg.hardware_type
        device.device_info = device_reg.device_info
        device.contribution_weight = device_reg.contribution_weight
        device.region = device_reg.region
        device.status = "online"
        device.last_sync = datetime.now(timezone.utc)
        device.last_heartbeat = datetime.now(timezone.utc)
    else:
        device = Device(
            client_id=device_reg.client_id,
            name=f"Node-{device_reg.client_id.split('_')[-1]}",
            hardware_type=device_reg.hardware_type,
            status="online",
            cpu_usage=0.0,
            memory_usage=0.0,
            last_sync=datetime.now(timezone.utc),
            contribution_score=100.0,
            region=device_reg.region,
            device_info=device_reg.device_info,
            contribution_weight=device_reg.contribution_weight,
            last_heartbeat=datetime.now(timezone.utc)
        )
        db.add(device)
    
    await db.commit()
    await db.refresh(device)
    
    device_data = EdgeDevice(
        id=device.client_id,
        name=device.name,
        hardwareType=device.hardware_type,
        status=device.status,
        cpuUsage=device.cpu_usage,
        memoryUsage=device.memory_usage,
        lastSync=device.last_sync.isoformat() if device.last_sync else "",
        contributionScore=device.contribution_score,
        region=device.region
    )
    
    await manager.broadcast("devices", {
        "event": "device.registered",
        "data": device_data.model_dump()
    })
    
    return device_data

@router.post("/{client_id}/heartbeat")
async def device_heartbeat(client_id: str, heartbeat: DeviceHeartbeat, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Device).where(Device.client_id == client_id))
    device = result.scalars().first()
    
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
        
    device.status = heartbeat.status
    device.cpu_usage = heartbeat.cpu_usage
    device.memory_usage = heartbeat.memory_usage
    device.last_heartbeat = datetime.now(timezone.utc)
    
    await db.commit()
    
    await manager.broadcast("devices", {
        "event": "device.heartbeat",
        "data": {
            "id": device.client_id,
            "status": device.status,
            "cpuUsage": device.cpu_usage,
            "memoryUsage": device.memory_usage
        }
    })
    
    return {"status": "ok"}

@router.get("", response_model=List[EdgeDevice])
async def get_devices(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Device))
    devices = result.scalars().all()
    
    return [
        EdgeDevice(
            id=d.client_id,
            name=d.name,
            hardwareType=d.hardware_type,
            status=d.status,
            cpuUsage=d.cpu_usage,
            memoryUsage=d.memory_usage,
            lastSync=d.last_sync.isoformat() if d.last_sync else "",
            contributionScore=d.contribution_score,
            region=d.region
        ) for d in devices
    ]

@router.get("/regions", response_model=List[RegionData])
async def get_regions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Device.region, func.count(Device.id)).group_by(Device.region))
    regions_count = result.all()
    
    # Mock coordinates for regions for the network map
    region_coords = {
        "US-East": {"x": 25, "y": 30},
        "US-West": {"x": 15, "y": 35},
        "EU-Central": {"x": 48, "y": 25},
        "AP-East": {"x": 75, "y": 35},
        "SA-East": {"x": 30, "y": 60}
    }
    
    return [
        RegionData(
            name=r[0] or "Unknown",
            deviceCount=r[1],
            x=region_coords.get(r[0], {"x": 50})["x"],
            y=region_coords.get(r[0], {"y": 50})["y"]
        ) for r in regions_count
    ]
