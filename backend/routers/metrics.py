from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Dict

from backend.database import get_db
from backend.models import Metric, Round, Device
from backend.schemas import ChartDataPoint, MetricCreate
from backend.websocket.manager import manager
from datetime import datetime, timezone

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

@router.post("", response_model=Dict[str, int])
async def create_metric(metric: MetricCreate, db: AsyncSession = Depends(get_db)):
    db_metric = Metric(
        client_id=metric.client_id,
        round_number=metric.round_number,
        epoch=metric.epoch,
        avg_loss=metric.avg_loss,
        accuracy=metric.accuracy,
        epsilon_spent=metric.epsilon_spent,
        delta=metric.delta,
        data_size=metric.data_size,
        partition_info=metric.partition_info,
        training_duration_s=metric.training_duration_s,
        cpu_usage=metric.cpu_usage,
        memory_usage=metric.memory_usage
    )
    db.add(db_metric)
    await db.commit()
    await db.refresh(db_metric)
    
    await manager.broadcast("metrics", {
        "event": "metric.reported",
        "data": {
            "client_id": db_metric.client_id,
            "round_number": db_metric.round_number,
            "epoch": db_metric.epoch,
            "avg_loss": db_metric.avg_loss,
            "accuracy": db_metric.accuracy,
            "epsilon_spent": db_metric.epsilon_spent
        }
    })
    
    return {"id": db_metric.id}

@router.get("/accuracy-trend", response_model=List[ChartDataPoint])
async def get_accuracy_trend(db: AsyncSession = Depends(get_db)):
    # Returns the average accuracy per round (mocking with a slight upward trend based on round number for now if no data)
    result = await db.execute(
        select(Metric.round_number, func.avg(Metric.accuracy))
        .where(Metric.accuracy.is_not(None))
        .group_by(Metric.round_number)
        .order_by(Metric.round_number.desc())
        .limit(8)
    )
    metrics = result.all()
    
    # Reverse to get chronological order
    metrics = list(reversed(metrics))
    
    if not metrics:
        return [
            ChartDataPoint(label="Round 1", value=92.5),
            ChartDataPoint(label="Round 2", value=93.1),
            ChartDataPoint(label="Round 3", value=93.8),
            ChartDataPoint(label="Round 4", value=94.2)
        ]
        
    return [
        ChartDataPoint(label=f"Round {m[0]}", value=round(m[1], 2)) for m in metrics
    ]

@router.get("/loss-curve", response_model=List[ChartDataPoint])
async def get_loss_curve(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Metric.round_number, func.avg(Metric.avg_loss))
        .where(Metric.avg_loss.is_not(None))
        .group_by(Metric.round_number)
        .order_by(Metric.round_number.desc())
        .limit(8)
    )
    metrics = result.all()
    metrics = list(reversed(metrics))
    
    if not metrics:
        return [
            ChartDataPoint(label="Round 1", value=0.15),
            ChartDataPoint(label="Round 2", value=0.12),
            ChartDataPoint(label="Round 3", value=0.09),
            ChartDataPoint(label="Round 4", value=0.07)
        ]
        
    return [
        ChartDataPoint(label=f"Round {m[0]}", value=round(m[1], 4)) for m in metrics
    ]

@router.get("/analytics-accuracy", response_model=List[ChartDataPoint])
async def get_analytics_accuracy(db: AsyncSession = Depends(get_db)):
    # Reuses accuracy trend logic for now
    return await get_accuracy_trend(db)

@router.get("/device-participation", response_model=List[ChartDataPoint])
async def get_device_participation(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Round.round_number, Round.received_clients)
        .order_by(Round.round_number.desc())
        .limit(8)
    )
    rounds = result.all()
    rounds = list(reversed(rounds))
    
    if not rounds:
        return [
            ChartDataPoint(label="W1", value=180),
            ChartDataPoint(label="W2", value=210),
            ChartDataPoint(label="W3", value=247)
        ]
        
    return [
        ChartDataPoint(label=f"R{r[0]}", value=r[1]) for r in rounds
    ]

@router.get("/training-throughput", response_model=List[ChartDataPoint])
async def get_training_throughput(db: AsyncSession = Depends(get_db)):
    # Mocking throughput data
    return [
        ChartDataPoint(label="Mon", value=12.4),
        ChartDataPoint(label="Tue", value=14.2),
        ChartDataPoint(label="Wed", value=15.8),
        ChartDataPoint(label="Thu", value=13.5),
        ChartDataPoint(label="Fri", value=18.2)
    ]

@router.get("/resource-utilization", response_model=List[ChartDataPoint])
async def get_resource_utilization(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(func.avg(Device.cpu_usage), func.avg(Device.memory_usage))
    )
    avg_cpu, avg_mem = result.first()
    
    return [
        ChartDataPoint(label="CPU", value=round(avg_cpu or 45.0, 1), value2=40.0),
        ChartDataPoint(label="Memory", value=round(avg_mem or 60.0, 1), value2=55.0),
        ChartDataPoint(label="Network", value=75.0, value2=65.0)
    ]
