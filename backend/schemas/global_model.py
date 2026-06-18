from pydantic import BaseModel

class GlobalModelSchema(BaseModel):
    name: str
    version: str
    accuracy: float
    lastUpdated: str

    class Config:
        from_attributes = True
