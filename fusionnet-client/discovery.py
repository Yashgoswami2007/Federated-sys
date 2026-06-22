"""
discovery.py — LAN/WiFi auto-discovery for FusionNet nodes.

Coordinator side: call advertise_coordinator() to broadcast the backend URL via mDNS.
Client side:      call find_coordinator() to scan the local network and return the URL.

Requires: zeroconf>=0.132.2  (pip install zeroconf)
"""

import logging
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_fusionnet._tcp.local."
SERVICE_NAME = "FusionNetCoordinator._fusionnet._tcp.local."


def _local_ip() -> str:
    """Best-effort local IP (not 127.0.0.1)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def advertise_coordinator(port: int = 8000) -> "ServiceRegistration":
    """
    Broadcast this machine as the FusionNet coordinator on the local network.
    Returns a ServiceRegistration object — call .stop() to deregister.

    Usage:
        reg = advertise_coordinator(port=8000)
        # ... run your server ...
        reg.stop()
    """
    from zeroconf import Zeroconf, ServiceInfo

    ip = _local_ip()
    logger.info(f"Advertising coordinator on {ip}:{port} via mDNS ({SERVICE_TYPE})")

    info = ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={b"version": b"1", b"path": b"/"},
        server=f"{socket.gethostname()}.local.",
    )

    zc = Zeroconf()
    zc.register_service(info)

    class ServiceRegistration:
        def __init__(self, zeroconf_instance, service_info):
            self._zc = zeroconf_instance
            self._info = service_info

        def stop(self):
            self._zc.unregister_service(self._info)
            self._zc.close()
            logger.info("mDNS coordinator advertisement stopped.")

    return ServiceRegistration(zc, info)


def find_coordinator(
    timeout: float = 10.0,
    fallback_url: Optional[str] = None,
) -> Optional[str]:
    """
    Scan the local network for a FusionNet coordinator advertised via mDNS.
    Returns the backend URL string (e.g. "http://192.168.1.42:8000") or
    fallback_url if nothing is found within timeout seconds.

    Usage:
        url = find_coordinator(timeout=10, fallback_url="http://localhost:8000")
    """
    try:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange
    except ImportError:
        logger.warning("zeroconf not installed — skipping LAN discovery. pip install zeroconf")
        return fallback_url

    found_url: list[str] = []  # use list so closure can write to it

    def on_service_state_change(zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                port = info.port
                url = f"http://{ip}:{port}"
                logger.info(f"Discovered coordinator at {url} (host: {name})")
                found_url.append(url)

    zc = Zeroconf()
    browser = ServiceBrowser(zc, SERVICE_TYPE, handlers=[on_service_state_change])

    deadline = time.time() + timeout
    while not found_url and time.time() < deadline:
        time.sleep(0.2)

    browser.cancel()
    zc.close()

    if found_url:
        return found_url[0]

    if fallback_url:
        logger.info(f"No coordinator found on LAN after {timeout}s — using fallback: {fallback_url}")
        return fallback_url

    logger.warning(f"No coordinator found on LAN after {timeout}s and no fallback configured.")
    return None
