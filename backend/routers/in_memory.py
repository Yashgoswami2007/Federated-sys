from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException


router = APIRouter()

devices: dict[str, dict[str, Any]] = {}
rounds: dict[int, dict[str, Any]] = {}
metrics: list[dict[str, Any]] = []
events: list[dict[str, Any]] = []
global_models: list[dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_training_job(round_data: dict[str, Any]) -> dict[str, Any]:
    round_number = round_data["round_number"]
    return {
        "id": f"round_{round_number}",
        "round": round_number,
        "totalRounds": round_data.get("total_rounds", 1),
        "progress": round_data.get("progress", 0),
        "estimatedCompletion": "Calculating..." if round_data.get("status") == "running" else "",
        "participatingDevices": round_data.get("expected_clients", 0),
        "modelVersion": round_data.get("model_version", "unknown"),
        "status": round_data.get("status", "pending"),
    }


def to_activity(event: dict[str, Any]) -> dict[str, Any]:
    type_map = {
        "device.registered": "device_joined",
        "round.completed": "round_completed",
        "global.update_published": "model_updated",
        "model.global_updated": "model_updated",
        "security.verified": "security_verified",
    }
    return {
        "id": f"evt_{event['id']}",
        "type": type_map.get(event["event_type"], "round_completed"),
        "message": event["message"],
        "timestamp": event["created_at"],
    }


@router.post("/api/devices/register")
async def register_device(payload: dict[str, Any]):
    client_id = payload["client_id"]
    device = {
        "id": client_id,
        "name": f"Node-{client_id.split('_')[-1]}",
        "hardwareType": payload.get("hardware_type", "Unknown"),
        "status": "online",
        "cpuUsage": 0.0,
        "memoryUsage": 0.0,
        "lastSync": now_iso(),
        "contributionScore": 100.0,
        "region": payload.get("region", "Local"),
        "deviceInfo": payload.get("device_info", {}),
        "contributionWeight": payload.get("contribution_weight", 1.0),
    }
    devices[client_id] = device
    return device


@router.get("/api/devices")
async def get_devices():
    return list(devices.values())


@router.get("/api/devices/regions")
async def get_regions():
    region_counts: dict[str, int] = {}
    for device in devices.values():
        region = device.get("region") or "Local"
        region_counts[region] = region_counts.get(region, 0) + 1
    return [
        {"name": region, "deviceCount": count, "x": 50, "y": 50}
        for region, count in region_counts.items()
    ]


@router.post("/api/rounds")
async def create_round(payload: dict[str, Any]):
    round_number = int(payload["round_number"])
    round_data = {
        "round_number": round_number,
        "total_rounds": payload.get("total_rounds", 1),
        "expected_clients": payload["expected_clients"],
        "received_clients": 0,
        "progress": 0,
        "status": "running",
        "model_version": payload.get("model_version", "local-mvp"),
        "started_at": now_iso(),
        "completed_at": None,
        "global_model_path": None,
    }
    rounds[round_number] = round_data
    return to_training_job(round_data)


@router.patch("/api/rounds/{round_number}")
async def update_round(round_number: int, payload: dict[str, Any]):
    if round_number not in rounds:
        raise HTTPException(status_code=404, detail="Round not found")
    round_data = rounds[round_number]
    round_data.update({key: value for key, value in payload.items() if value is not None})
    if round_data.get("status") in {"completed", "failed"} and not round_data.get("completed_at"):
        round_data["completed_at"] = now_iso()
    return to_training_job(round_data)


@router.get("/api/rounds/jobs")
async def get_jobs():
    return [to_training_job(item) for item in sorted(rounds.values(), key=lambda r: r["round_number"], reverse=True)]


@router.get("/api/rounds/current")
async def get_current_round():
    if not rounds:
        raise HTTPException(status_code=404, detail="No rounds found")
    latest = max(rounds)
    return to_training_job(rounds[latest])


@router.post("/api/metrics")
async def create_metric(payload: dict[str, Any]):
    metric = {"id": len(metrics) + 1, **payload, "reported_at": now_iso()}
    metrics.append(metric)
    return {"id": metric["id"]}


@router.get("/api/metrics/accuracy-trend")
@router.get("/api/metrics/analytics-accuracy")
async def get_accuracy_trend():
    by_round: dict[int, list[float]] = {}
    for metric in metrics:
        if metric.get("accuracy") is not None:
            by_round.setdefault(int(metric["round_number"]), []).append(float(metric["accuracy"]))
    return [
        {"label": f"Round {round_num}", "value": round(sum(values) / len(values), 4)}
        for round_num, values in sorted(by_round.items())
    ]


@router.get("/api/metrics/loss-curve")
async def get_loss_curve():
    by_round: dict[int, list[float]] = {}
    for metric in metrics:
        if metric.get("avg_loss") is not None:
            by_round.setdefault(int(metric["round_number"]), []).append(float(metric["avg_loss"]))
    return [
        {"label": f"Round {round_num}", "value": round(sum(values) / len(values), 6)}
        for round_num, values in sorted(by_round.items())
    ]


@router.get("/api/metrics/device-participation")
async def get_device_participation():
    return [
        {"label": f"R{round_num}", "value": item.get("received_clients", 0)}
        for round_num, item in sorted(rounds.items())
    ]


@router.get("/api/metrics/training-throughput")
async def get_training_throughput():
    return []


@router.get("/api/metrics/resource-utilization")
async def get_resource_utilization():
    return []


@router.post("/api/events")
async def create_event(payload: dict[str, Any]):
    event = {"id": len(events) + 1, **payload, "created_at": now_iso()}
    events.append(event)
    return {"id": event["id"]}


@router.get("/api/events/activity")
async def get_activity_feed():
    return [to_activity(event) for event in reversed(events[-10:])]


@router.get("/api/dashboard/kpi")
async def get_kpi_metrics():
    completed_rounds = len([item for item in rounds.values() if item.get("status") == "completed"])
    latest_accuracy = None
    for metric in reversed(metrics):
        if metric.get("accuracy") is not None:
            latest_accuracy = float(metric["accuracy"])
            break
    return [
        {"label": "Active Devices", "value": str(len(devices)), "change": "live", "trend": "neutral", "icon": "cpu"},
        {"label": "Training Rounds", "value": str(completed_rounds), "change": "live", "trend": "neutral", "icon": "layers"},
        {
            "label": "Model Accuracy",
            "value": f"{latest_accuracy * 100:.2f}%" if latest_accuracy is not None else "—",
            "change": "live",
            "trend": "neutral",
            "icon": "target",
        },
        {"label": "Security Score", "value": "—", "change": "local", "trend": "neutral", "icon": "shield"},
    ]


@router.post("/api/models/global")
async def create_global_model(round_number: int, hf_path: str, payload: dict[str, Any]):
    model = {
        **payload,
        "round_number": round_number,
        "hf_path": hf_path,
        "lastUpdated": now_iso(),
    }
    global_models.append(model)
    return model


@router.get("/api/models/global")
async def get_global_model():
    if not global_models:
        return {
            "name": "FusionNet Local MVP Adapter",
            "version": "local-mvp",
            "accuracy": 0.0,
            "lastUpdated": "",
        }
    return global_models[-1]
