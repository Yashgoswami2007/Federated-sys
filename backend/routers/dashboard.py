from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from backend.database import get_db
from backend.models import Device, Round, Metric
from backend.schemas import KpiMetric

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/kpi", response_model=List[KpiMetric])
async def get_kpi_metrics(db: AsyncSession = Depends(get_db)):
    # Calculate Active Devices
    active_result = await db.execute(select(func.count(Device.id)).where(Device.status != "offline"))
    active_devices = active_result.scalar() or 0
    
    # Calculate Training Rounds
    rounds_result = await db.execute(select(func.count(Round.id)).where(Round.status == "completed"))
    completed_rounds = rounds_result.scalar() or 0
    
    # Calculate Latest Model Accuracy
    accuracy_result = await db.execute(
        select(Metric.accuracy)
        .where(Metric.accuracy.is_not(None))
        .order_by(Metric.round_number.desc(), Metric.epoch.desc())
        .limit(1)
    )
    latest_accuracy = accuracy_result.scalar()
    
    return [
        KpiMetric(
            label="Active Devices",
            value=str(active_devices),
            change="+12 from last round",
            trend="up",
            icon="cpu"
        ),
        KpiMetric(
            label="Training Rounds",
            value=str(completed_rounds),
            change="+1 this week",
            trend="up",
            icon="layers"
        ),
        KpiMetric(
            label="Model Accuracy",
            value=f"{round(latest_accuracy, 2)}%" if latest_accuracy else "0.0%",
            change="+0.8% overall",
            trend="up",
            icon="target"
        ),
        KpiMetric(
            label="Security Score",
            value="98",
            change="Zero incidents",
            trend="neutral",
            icon="shield"
        )
    ]
