from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChartDataPoint(BaseModel):
    label: str
    value: float
    value2: Optional[float] = None

class MetricCreate(BaseModel):
    client_id: str
    round_number: int
    epoch: Optional[int] = None
    avg_loss: Optional[float] = None
    accuracy: Optional[float] = None
    epsilon_spent: Optional[float] = None
    delta: Optional[float] = None
    data_size: Optional[int] = None
    partition_info: Optional[Dict[str, Any]] = None
    training_duration_s: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
