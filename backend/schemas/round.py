from pydantic import BaseModel
from typing import Optional

class TrainingJob(BaseModel):
    id: str
    round: int
    totalRounds: int
    progress: int
    estimatedCompletion: str
    participatingDevices: int
    modelVersion: str
    status: str

    class Config:
        from_attributes = True

class RoundCreate(BaseModel):
    round_number: int
    total_rounds: int
    expected_clients: int
    model_version: str

class RoundUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    received_clients: Optional[int] = None
    global_model_path: Optional[str] = None
