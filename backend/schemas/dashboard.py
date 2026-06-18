from pydantic import BaseModel

class KpiMetric(BaseModel):
    label: str
    value: str
    change: str
    trend: str
    icon: str
