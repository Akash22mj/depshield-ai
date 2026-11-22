import json
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from depshield.autonomous_agent import AutonomousSupplyChainAgent, ManifestAuditSummary
from depshield.dataset_builder import calculate_shannon_entropy
from depshield.ensemble_judge import EnsembleAuditReport, EnsembleJudge
from depshield.interactive_features import (
    AskShieldRequest,
    AskShieldResponse,
    HallucinationHuntResponse,
    PyPIFeedItem,
    RedTeamPatchRequest,
    RedTeamResult,
    RedTeamSubmission,
    ShowdownMoveRequest,
    ShowdownMoveResponse,
    ask_shield_analyst,
    get_showdown_leaderboard,
    hunt_llm_hallucinations,
    load_redteam_hall_of_fame,
    patch_redteam_signature,
    run_showdown_move,
    scan_live_pypi_feed,
    submit_redteam_attempt,
)
from depshield.schemas import PackageRecord

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_HTML = PROJECT_ROOT / "frontend" / "index.html"
TSNE_CACHE_PATH = PROJECT_ROOT / "data" / "vectorstore" / "tsne_map.json"
BENCHMARK_JSON_PATH = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
SCALED_BENCHMARK_JSON_PATH = PROJECT_ROOT / "evaluation" / "scaled_benchmark_results.json"
RIGOROUS_BENCHMARK_JSON_PATH = PROJECT_ROOT / "evaluation" / "rigorous_benchmark_results.json"

app = FastAPI(
    title="DepShield AI Security API",
    description="Orthogonal 4-Stage Supply-Chain Firewall REST API",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JUDGE = EnsembleJudge(use_modal=True)
CPU_JUDGE = EnsembleJudge(use_modal=False)
MANIFEST_AGENT = AutonomousSupplyChainAgent(use_modal=False)

_IP_REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_MAX_CALLS = 35
RATE_LIMIT_WINDOW_SEC = 3600


def _enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if client_ip in {"127.0.0.1", "::1", "localhost"}:
        return
    now = time.time()
    window = _IP_REQUEST_LOG[client_ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SEC:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX_CALLS:
        raise HTTPException(
            status_code=429,
            detail="Public demo rate limit reached (35 LLM scans/hour per IP).",
        )
    window.append(now)


class PackageAuditRequest(BaseModel):
    package_name: str = Field(..., max_length=120)
    version: str = Field(default="0.1.0", max_length=40)
    age_days: float = 1.0
    release_count: int = 1
    summary: str = Field(default="", max_length=500)
    setup_code: str = Field(default="", max_length=12000)
    use_modal_gpu: bool = True


class ManifestAuditRequest(BaseModel):
    requirements_txt: str = Field(..., max_length=8000)
    send_pushover: bool = True


class HuntRequest(BaseModel):
    task_prompt: str = Field(..., max_length=1500)


@app.get("/")
def serve_3d_frontend():
    return FileResponse(FRONTEND_HTML)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "online", "service": "DepShield AI", "stages": 4, "zero_execution": True}


@app.post("/api/audit", response_model=EnsembleAuditReport)
def audit_package_endpoint(req: PackageAuditRequest, request: Request) -> EnsembleAuditReport:
    _enforce_rate_limit(request)
    clean_code = req.setup_code.strip() or (
        f"from setuptools import setup\nsetup(name='{req.package_name}', version='{req.version}')\n"
    )
    record = PackageRecord(
        package_name=req.package_name.strip(),
        version=req.version.strip(),
        label="unknown",
        source="rest_api",
        summary=req.summary[:500],
        setup_code=clean_code,
        age_days=float(max(0.0, req.age_days)),
        release_count=int(max(1, req.release_count)),
        code_size_bytes=len(clean_code),
        shannon_entropy=calculate_shannon_entropy(clean_code),
    )
    JUDGE.use_modal = req.use_modal_gpu
    return JUDGE.audit_package(record, check_live_registry=True)


@app.post("/api/manifest", response_model=ManifestAuditSummary)
def audit_manifest_endpoint(req: ManifestAuditRequest, request: Request) -> ManifestAuditSummary:
    _enforce_rate_limit(request)
    return MANIFEST_AGENT.audit_manifest(
        requirements_text=req.requirements_txt,
        send_pushover=req.send_pushover,
    )


@app.post("/api/hunt", response_model=HallucinationHuntResponse)
def hallucination_hunter_endpoint(req: HuntRequest, request: Request) -> HallucinationHuntResponse:
    _enforce_rate_limit(request)
    return hunt_llm_hallucinations(req.task_prompt.strip(), CPU_JUDGE)


@app.post("/api/ask-shield", response_model=AskShieldResponse)
def ask_shield_endpoint(req: AskShieldRequest, request: Request) -> AskShieldResponse:
    _enforce_rate_limit(request)
    return ask_shield_analyst(req, CPU_JUDGE)


@app.post("/api/showdown/move", response_model=ShowdownMoveResponse)
def showdown_move_endpoint(req: ShowdownMoveRequest, request: Request) -> ShowdownMoveResponse:
    _enforce_rate_limit(request)
    return run_showdown_move(req, CPU_JUDGE)


@app.get("/api/showdown/leaderboard")
def showdown_leaderboard_endpoint() -> dict:
    return get_showdown_leaderboard()


@app.post("/api/redteam", response_model=RedTeamResult)
def redteam_submit_endpoint(sub: RedTeamSubmission, request: Request) -> RedTeamResult:
    _enforce_rate_limit(request)
    return submit_redteam_attempt(sub, CPU_JUDGE)


@app.get("/api/redteam/hall-of-fame")
def redteam_hall_of_fame_endpoint() -> list[dict]:
    return load_redteam_hall_of_fame(limit=20)


@app.post("/api/redteam/patch")
def redteam_patch_endpoint(req: RedTeamPatchRequest, request: Request) -> dict:
    _enforce_rate_limit(request)
    return patch_redteam_signature(req)


@app.get("/api/pypi-feed", response_model=list[PyPIFeedItem])
def pypi_live_feed_endpoint(request: Request) -> list[PyPIFeedItem]:
    _enforce_rate_limit(request)
    return scan_live_pypi_feed(CPU_JUDGE, limit=6)


@app.get("/api/benchmarks")
def get_benchmarks() -> dict:
    for path in (RIGOROUS_BENCHMARK_JSON_PATH, SCALED_BENCHMARK_JSON_PATH, BENCHMARK_JSON_PATH):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"ablation_table": [], "adversarial_stress_test": []}


@app.get("/api/tsne")
def get_tsne_map() -> list[dict]:
    if not TSNE_CACHE_PATH.exists():
        return []
    return json.loads(TSNE_CACHE_PATH.read_text(encoding="utf-8"))