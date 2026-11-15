# 🚀 05 — Deployment, Operations & Open-Core Licensing Manual
### **Local Dev, Docker, Kubernetes & Production SLA Playbook**

---

## 1. Local Development Workflow

### Backend & Quantum Engine
```bash
# 1. Create and activate Python 3.11+ virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install PravahaQ in editable mode with all Quantum SDKs and Test tools
pip install -e ".[dev,quantum]"

# 3. Verify 100% of Unit, Integration, and Property tests pass
pytest tests/ -v

# 4. Start the FastAPI Control Plane
uvicorn api.main:app --reload --port 8000
```

### Frontend ("Quantum Obsidian" Mission Control)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173 in your browser
```

---

## 2. Production Docker & Kubernetes Deployment

### Docker Compose (Single-Node Production)
```bash
docker compose up --build -d
# API Health Check: curl http://localhost:8000/api/v1/health
# Frontend UI:      http://localhost:5173
```

### Kubernetes Cluster Deployment
```bash
kubectl apply -f deploy/k8s/deployment-api.yaml
kubectl apply -f deploy/k8s/service-and-ingress.yaml
```

---

## 3. Production SLA Fallback Architecture

In live edge-microgrid control loops, the control plane enforces a **strict 450ms SLA timeout**:
1. **Fast Path (<50ms):** Checks SHA-256 content-addressable cache (`infrastructure/content_addressable_cache.py`) for isomorphic HUBO tensors.
2. **Primary Path (50–350ms):** Executes `BF-DCQO` or `PCE` on the configured quantum simulator/QPU session or `MPS` Tensor Network surrogate.
3. **Guaranteed Fallback (350–450ms):** If a cloud QPU queue spikes, the `AsyncJobFSM` automatically falls back to the **MPS Tensor Network Surrogate + Deterministic Feasibility Repair**, ensuring zero dropped infrastructure frames.
