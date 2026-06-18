from pydantic import BaseModel
from typing import Optional, Dict, Any

class EdgeDevice(BaseModel):
    id: str
    name: str
    hardwareType: str
    status: str
    cpuUsage: float
    memoryUsage: float
    lastSync: str
    contributionScore: float
    region: str

    class Config:
        from_attributes = True

class RegionData(BaseModel):
    name: str
    deviceCount: int
    x: int
    y: int

class DeviceRegister(BaseModel):
    client_id: str
    hardware_type: str
    device_info: Dict[str, Any]
    contribution_weight: float = 1.0
    region: Optional[str] = "US-East"

class DeviceHeartbeat(BaseModel):
    status: str
    cpu_usage: float
    memory_usage: float
