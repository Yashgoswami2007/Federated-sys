from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from backend.database import get_db
from backend.models import Round
from backend.schemas import TrainingJob, RoundCreate, RoundUpdate
from backend.websocket.manager import manager
from datetime import datetime, timezone

router = APIRouter(prefix="/api/rounds", tags=["rounds"])

@router.post("", response_model=TrainingJob)
async def create_round(round_create: RoundCreate, db: AsyncSession = Depends(get_db)):
    db_round = Round(
        round_number=round_create.round_number,
        total_rounds=round_create.total_rounds,
        expected_clients=round_create.expected_clients,
        model_version=round_create.model_version,
        status="running",
        started_at=datetime.now(timezone.utc)
    )
    db.add(db_round)
    await db.commit()
    await db.refresh(db_round)
    
    job_data = TrainingJob(
        id=f"round_{db_round.round_number}",
        round=db_round.round_number,
        totalRounds=db_round.total_rounds,
        progress=0,
        estimatedCompletion="Calculating...",
        participatingDevices=db_round.expected_clients,
        modelVersion=db_round.model_version or "unknown",
        status=db_round.status
    )
    
    await manager.broadcast("rounds", {
        "event": "round.created",
        "data": job_data.model_dump()
    })
    
    return job_data

@router.patch("/{round_number}", response_model=TrainingJob)
async def update_round(round_number: int, round_update: RoundUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Round).where(Round.round_number == round_number))
    db_round = result.scalars().first()
    
    if not db_round:
        raise HTTPException(status_code=404, detail="Round not found")
        
    if round_update.status:
        db_round.status = round_update.status
        if round_update.status in ["completed", "failed"]:
            db_round.completed_at = datetime.now(timezone.utc)
    if round_update.progress is not None:
        db_round.progress = round_update.progress
    if round_update.received_clients is not None:
        db_round.received_clients = round_update.received_clients
    if round_update.global_model_path:
        db_round.global_model_path = round_update.global_model_path
        
    await db.commit()
    await db.refresh(db_round)
    
    job_data = TrainingJob(
        id=f"round_{db_round.round_number}",
        round=db_round.round_number,
        totalRounds=db_round.total_rounds,
        progress=db_round.progress,
        estimatedCompletion="Calculating..." if db_round.status == "running" else "",
        participatingDevices=db_round.expected_clients,
        modelVersion=db_round.model_version or "unknown",
        status=db_round.status
    )
    
    await manager.broadcast("rounds", {
        "event": "round.updated",
        "data": job_data.model_dump()
    })
    
    return job_data

@router.get("/jobs", response_model=List[TrainingJob])
async def get_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Round).order_by(Round.round_number.desc()))
    rounds = result.scalars().all()
    
    return [
        TrainingJob(
            id=f"round_{r.round_number}",
            round=r.round_number,
            totalRounds=r.total_rounds,
            progress=r.progress,
            estimatedCompletion="Calculating..." if r.status == "running" else "",
            participatingDevices=r.expected_clients,
            modelVersion=r.model_version or "unknown",
            status=r.status
        ) for r in rounds
    ]

@router.get("/current", response_model=TrainingJob)
async def get_current_round(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Round).order_by(Round.round_number.desc()).limit(1))
    current = result.scalars().first()
    
    if not current:
        raise HTTPException(status_code=404, detail="No rounds found")
        
    return TrainingJob(
        id=f"round_{current.round_number}",
        round=current.round_number,
        totalRounds=current.total_rounds,
        progress=current.progress,
        estimatedCompletion="Calculating..." if current.status == "running" else "",
        participatingDevices=current.expected_clients,
        modelVersion=current.model_version or "unknown",
        status=current.status
    )
