from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class PackageRecord(BaseModel):
    """Canonical record for both training/eval samples and live PyPI packages."""

    package_name: str = Field(description="Exact PyPI package name")
    version: str = Field(default="0.0.0", description="Package version inspected")
    label: Literal["malicious", "benign", "unknown"] = Field(
        default="unknown", description="Ground-truth label for evaluation datasets"
    )
    source: str = Field(
        description="Origin: 'openssf', 'datadog', 'pypi_clean', or 'live_mcp'"
    )
    summary: str = Field(default="", description="Package summary or advisory description")
    setup_code: str = Field(
        default="", description="Extracted setup.py / install hook / __init__.py snippet"
    )
    age_days: float = Field(default=0.0, description="Days since first PyPI release")
    release_count: int = Field(default=1, description="Total number of published versions")
    code_size_bytes: int = Field(default=0, description="Size of inspected source code")
    shannon_entropy: float = Field(
        default=0.0, description="Shannon entropy of code (high > 5.2 often indicates obfuscation)"
    )
    cwe_ids: list[str] = Field(
        default_factory=list, description="Associated CWE identifiers if known"
    )


class StaticScanResult(BaseModel):
    """Stage 1 deterministic AST + PyPI registry baseline verdict."""

    package_name: str
    exists_on_pypi: bool = Field(description="False indicates potential slopsquatting")
    typosquat_target: str | None = Field(
        default=None, description="Popular package name if edit distance is 1-2"
    )
    suspicious_imports: list[str] = Field(
        default_factory=list, description="Risky modules like subprocess, socket, ctypes"
    )
    suspicious_calls: list[str] = Field(
        default_factory=list, description="Risky calls like eval, exec, os.system, base64.b64decode"
    )
    has_install_hook: bool = Field(
        default=False, description="True if setup.py overrides cmdclass install/develop"
    )
    static_risk_score: float = Field(
        ge=0.0, le=1.0, description="Deterministic baseline risk probability (0.0 to 1.0)"
    )
    reasons: list[str] = Field(default_factory=list)


class ModelSignal(BaseModel):
    """Individual model output inside the 3-Model Hybrid Ensemble."""

    model_name: str = Field(description="e.g., 'rag_gemini', 'pytorch_tabular', 'modal_qlora'")
    risk_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    attack_category: Literal[
        "benign",
        "credential_stealer",
        "obfuscated_dropper",
        "reverse_shell",
        "slopsquatting_typosquat",
        "suspicious_unknown",
    ]
    reasoning: str = Field(description="Concise technical justification for the score")


class EnsembleVerdict(BaseModel):
    """Stage 2 calibrated decision blending RAG + PyTorch + Modal QLoRA."""

    package_name: str
    blended_risk_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["ALLOW", "SANDBOX_DETONATE", "BLOCK"]
    primary_category: str
    signals: list[ModelSignal]
    similar_known_attacks: list[str] = Field(
        default_factory=list, description="Top matching malware IDs from ChromaDB"
    )
    tsne_coords: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="[x, y, z] coordinates for 3D threat cluster visualization",
    )


class SandboxTripwireReport(BaseModel):
    """Stage 3 behavioral telemetry from disposable Docker detonation."""

    executed: bool = False
    secret_file_read: bool = Field(default=False, description="Tripwire 1: Accessed .env / .ssh")
    outbound_network_call: bool = Field(default=False, description="Tripwire 2: Socket/DNS call")
    process_spawned: bool = Field(default=False, description="Tripwire 3: subprocess / shell")
    persistence_write: bool = Field(default=False, description="Tripwire 4: Wrote outside /tmp")
    obfuscated_runtime_exec: bool = Field(default=False, description="Tripwire 5: Dynamic exec/eval")
    triggered_behaviors: list[str] = Field(default_factory=list)
    raw_trace: str = ""


class FinalSecurityVerdict(BaseModel):
    """Top-level auditable output returned to MCP coding agents and Gradio UI."""

    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    package_name: str
    final_action: Literal["ALLOWED", "BLOCKED"]
    final_risk_score: float = Field(ge=0.0, le=1.0)
    static_scan: StaticScanResult
    ensemble_verdict: EnsembleVerdict
    sandbox_report: SandboxTripwireReport
    executive_summary: str