"""
load_balancer.py  –  XAI-Lab round-robin load balancer
-------------------------------------------------------
Distributes incoming requests across two backend services:
  • credit-backend-1  (model trained on credit_dataset1.csv)
  • credit-backend-2  (model trained on credit_dataset2.csv)

The backend URLs are resolved via Kubernetes ClusterIP service
names so no hard-coded IPs are needed.

Environment variables (optional overrides for local testing):
    BACKEND_1_URL  – default http://credit-backend-service-1:8000
    BACKEND_2_URL  – default http://credit-backend-service-2:8000
"""

import itertools
import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Backend pool ───────────────────────────────────────────────────────────────
# In Kubernetes these resolve to the ClusterIP services defined in the YAML.
# Override with env-vars for local/docker-compose testing.
BACKEND_SERVERS = [
    os.environ.get("BACKEND_1_URL", "http://credit-backend-service-1:8000"),
    os.environ.get("BACKEND_2_URL", "http://credit-backend-service-2:8000"),
]

server_pool = itertools.cycle(BACKEND_SERVERS)

FORWARD_TIMEOUT = int(os.environ.get("FORWARD_TIMEOUT_SECS", "10"))


# ── Helper ─────────────────────────────────────────────────────────────────────
def _next_backend() -> str:
    return next(server_pool)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/model-info", methods=["GET"])
def model_info():
    backend_url = _next_backend()
    try:
        resp = requests.get(f"{backend_url}/model-info", timeout=FORWARD_TIMEOUT)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc), "backend": backend_url}), 502


@app.route("/predict", methods=["POST"])
def predict():
    backend_url = _next_backend()
    try:
        resp = requests.post(
            f"{backend_url}/predict",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=FORWARD_TIMEOUT,
        )
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc), "backend": backend_url}), 502


@app.route("/health", methods=["GET"])
def health():
    """Probe both backends and return aggregated health."""
    results = {}
    for url in BACKEND_SERVERS:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            results[url] = r.json()
        except Exception as exc:
            results[url] = {"error": str(exc)}
    overall = "ok" if all("error" not in v for v in results.values()) else "degraded"
    return jsonify({"status": overall, "backends": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
