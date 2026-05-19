"""Production API client with persistent session, retry, and health monitoring."""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry


class APIClient:
    """Centralized API client with connection pooling, retry, and health tracking."""

    SUPPORTED_API_VERSION = "0.1.0"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = self._create_session()
        self._health_cache: dict[str, Any] | None = None
        self._health_cache_time: float = 0
        self._metrics_cache: dict[str, Any] | None = None
        self._metrics_cache_time: float = 0
        self._logs_cache: dict[str, Any] | None = None
        self._logs_cache_time: float = 0

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _is_cache_valid(self, cache_time: float, ttl: float) -> bool:
        return time.time() - cache_time < ttl

    def close(self) -> None:
        self._session.close()

    def health(self, use_cache: bool = True) -> dict[str, Any]:
        if use_cache and self._health_cache and self._is_cache_valid(self._health_cache_time, ttl=1.0):
            return self._health_cache

        try:
            response = self._session.get(f"{self.base_url}/health", timeout=2)
            response.raise_for_status()
            self._health_cache = response.json()
            self._health_cache_time = time.time()
            return self._health_cache
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}

    def get_metrics(self, use_cache: bool = True) -> dict[str, Any]:
        if use_cache and self._metrics_cache and self._is_cache_valid(self._metrics_cache_time, ttl=2.0):
            return self._metrics_cache

        try:
            response = self._session.get(f"{self.base_url}/observability/metrics", timeout=5)
            response.raise_for_status()
            self._metrics_cache = response.json()
            self._metrics_cache_time = time.time()
            return self._metrics_cache
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_diagnostics(self) -> dict[str, Any]:
        try:
            response = self._session.get(f"{self.base_url}/observability/diagnostics", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_logs(self, limit: int = 80, use_cache: bool = True) -> dict[str, Any]:
        if use_cache and self._logs_cache and self._is_cache_valid(self._logs_cache_time, ttl=2.0):
            return self._logs_cache

        try:
            response = self._session.get(
                f"{self.base_url}/observability/logs/recent",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            self._logs_cache = response.json()
            self._logs_cache_time = time.time()
            return self._logs_cache
        except requests.RequestException as e:
            return {"error": str(e)}

    def upload_video(self, video_file: tuple[str, bytes, str]) -> dict[str, Any]:
        files = {"video": video_file}
        try:
            response = self._session.post(
                f"{self.base_url}/videos/upload-and-process",
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def process_video(self, video_relative_path: str) -> dict[str, Any]:
        try:
            response = self._session.post(
                f"{self.base_url}/videos/process/async",
                json={"video_relative_path": video_relative_path},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_preview(self, job_id: str) -> dict[str, Any]:
        try:
            response = self._session.get(
                f"{self.base_url}/videos/processing-preview/{job_id}",
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_detections(self, limit: int = 100) -> dict[str, Any]:
        try:
            response = self._session.get(
                f"{self.base_url}/detections",
                params={"limit": limit},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def enroll_person(
        self, display_name: str, image_file: tuple[str, bytes, str], notes: str = ""
    ) -> dict[str, Any]:
        files = {"image": image_file}
        data = {"display_name": display_name, "notes": notes}
        try:
            response = self._session.post(
                f"{self.base_url}/persons",
                files=files,
                data=data,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def test_match(self, image_file: tuple[str, bytes, str]) -> dict[str, Any]:
        files = {"image": image_file}
        try:
            response = self._session.post(
                f"{self.base_url}/test-match",
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_video_status(self, job_id: str) -> dict[str, Any]:
        try:
            response = self._session.get(
                f"{self.base_url}/videos/processing-status/{job_id}",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_rejected_faces(self, job_id: str, limit: int = 12) -> dict[str, Any]:
        try:
            response = self._session.get(
                f"{self.base_url}/videos/processing-rejected-faces/{job_id}",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_latency_ms(self) -> float:
        start = time.time()
        self.health(use_cache=False)
        return (time.time() - start) * 1000

    def validate_version(self) -> tuple[bool, str]:
        health_data = self.health(use_cache=False)
        if "error" in health_data:
            return False, "Backend unreachable"
        backend_version = health_data.get("version", "unknown")
        if backend_version != self.SUPPORTED_API_VERSION:
            return False, f"Version mismatch: backend={backend_version}, dashboard={self.SUPPORTED_API_VERSION}"
        return True, "OK"
