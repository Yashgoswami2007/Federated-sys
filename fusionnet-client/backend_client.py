import httpx
import logging
import threading
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BackendClient:
    """Reports status and metrics to the FusionNet backend server.
    All calls are fire-and-forget — server being down never blocks training."""
    
    def __init__(self, base_url: str, hf_token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.client = httpx.Client(base_url=self.base_url, headers=self.headers, timeout=5.0)
        self.enabled = True
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        
    def _safe_post(self, path: str, json_data: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            response = self.client.post(path, json=json_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Backend API call failed ({path}): {str(e)}")
            return None
            
    def _safe_patch(self, path: str, json_data: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            response = self.client.patch(path, json=json_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Backend API call failed ({path}): {str(e)}")
            return None

    def register(self, client_id: str, hardware_info: dict) -> None:
        self._safe_post("/api/devices/register", {
            "client_id": client_id,
            "hardware_type": hardware_info.get("hardware_tier", "Unknown"),
            "device_info": hardware_info,
            "contribution_weight": hardware_info.get("contribution_weight", 1.0)
        })

    def heartbeat(self, client_id: str, status: str, cpu_usage: float = 0.0, memory_usage: float = 0.0) -> None:
        self._safe_post(f"/api/devices/{client_id}/heartbeat", {
            "status": status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        })

    def start_heartbeat_loop(self, client_id: str, interval: int = 30) -> None:
        if not self.enabled:
            return
            
        def _loop():
            while not self._stop_heartbeat.is_set():
                try:
                    import psutil
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                except ImportError:
                    cpu, mem = 0.0, 0.0
                
                self.heartbeat(client_id, "training", cpu, mem)
                time.sleep(interval)
                
        self._heartbeat_thread = threading.Thread(target=_loop, daemon=True)
        self._heartbeat_thread.start()

    def report_metrics(self, client_id: str, round_num: int, epoch: int, metrics: dict) -> None:
        data = {
            "client_id": client_id,
            "round_number": round_num,
            "epoch": epoch,
        }
        data.update(metrics)
        self._safe_post("/api/metrics", data)

    def report_event(self, event_type: str, message: str, severity: str = "info", metadata: dict = None) -> None:
        self._safe_post("/api/events", {
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "metadata_info": metadata or {}
        })

    def create_round(self, round_num: int, expected_clients: int, model_version: str = "v0.7.0") -> None:
        self._safe_post("/api/rounds", {
            "round_number": round_num,
            "total_rounds": 10, # default max rounds or configurable
            "expected_clients": expected_clients,
            "model_version": model_version
        })
        
    def update_round(self, round_num: int, status: str = None, progress: int = None, received_clients: int = None, global_model_path: str = None) -> None:
        update_data = {}
        if status is not None: update_data["status"] = status
        if progress is not None: update_data["progress"] = progress
        if received_clients is not None: update_data["received_clients"] = received_clients
        if global_model_path is not None: update_data["global_model_path"] = global_model_path
        
        if update_data:
            self._safe_patch(f"/api/rounds/{round_num}", update_data)
            
    def update_global_model(self, name: str, version: str, round_num: int, hf_path: str, accuracy: float = 94.2) -> None:
        self._safe_post("/api/models/global", {
            "name": name,
            "version": version,
            "accuracy": accuracy,
            "round_number": round_num,
            "hf_path": hf_path
        })
