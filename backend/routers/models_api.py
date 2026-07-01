from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db
from backend.models import GlobalModel
from backend.schemas import GlobalModelSchema
from datetime import datetime, timezone

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/global", response_model=GlobalModelSchema)
async def get_global_model(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GlobalModel).order_by(GlobalModel.round_number.desc()).limit(1))
    model = result.scalars().first()
    
    if not model:
        # Return fallback for initial state
        return GlobalModelSchema(
            name="TinyLlama-1.1B-Chat-AFLoRA",
            version="v0.7.0",
            accuracy=0.0,
            lastUpdated=datetime.now(timezone.utc).isoformat()
        )
        
    return GlobalModelSchema(
        name=model.name,
        version=model.version,
        accuracy=model.accuracy or 0.0,
        lastUpdated=model.updated_at.isoformat() if model.updated_at else ""
    )

@router.post("/global", response_model=GlobalModelSchema)
async def create_global_model(model: GlobalModelSchema, round_number: int, hf_path: str, db: AsyncSession = Depends(get_db)):
    db_model = GlobalModel(
        name=model.name,
        version=model.version,
        accuracy=model.accuracy,
        round_number=round_number,
        hf_path=hf_path,
        updated_at=datetime.now(timezone.utc)
    )
    db.add(db_model)
    await db.commit()
    await db.refresh(db_model)
    
    return GlobalModelSchema(
        name=db_model.name,
        version=db_model.version,
        accuracy=db_model.accuracy or 0.0,
        lastUpdated=db_model.updated_at.isoformat() if db_model.updated_at else ""
    )
