import contextlib
import json
import sys

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP  # type: ignore

from depshield.autonomous_agent import AutonomousSupplyChainAgent
from depshield.dataset_builder import calculate_shannon_entropy
from depshield.ensemble_judge import EnsembleJudge
from depshield.schemas import PackageRecord

mcp = FastMCP("DepShield-AI-Firewall")

_JUDGE: EnsembleJudge | None = None
_MANIFEST_AGENT: AutonomousSupplyChainAgent | None = None


def _get_engines() -> tuple[EnsembleJudge, AutonomousSupplyChainAgent]:
    """Lazy-load AI engines with stdout redirected to stderr so MCP JSON-RPC stays clean."""
    global _JUDGE, _MANIFEST_AGENT
    with contextlib.redirect_stdout(sys.stderr):
        if _JUDGE is None:
            _JUDGE = EnsembleJudge(use_modal=False)
        if _MANIFEST_AGENT is None:
            _MANIFEST_AGENT = AutonomousSupplyChainAgent(use_modal=False)
    return _JUDGE, _MANIFEST_AGENT


@mcp.tool()
def audit_python_package(
    package_name: str,
    setup_code: str = "",
    age_days: float = 1.0,
    release_count: int = 1,
) -> str:
    """
    Audits a Python package or setup.py snippet BEFORE running pip install.
    Uses DepShield AI's orthogonal multi-model ensemble (AST + PyPI Registry +
    ChromaDB RAG + PyTorch TabularRiskNet) to detect malware, prompt injection,
    and AI slopsquatting. Returns JSON with verdict (BLOCK, WARN, or ALLOW).
    """
    judge, _ = _get_engines()
    clean_code = setup_code.strip() or (
        f"from setuptools import setup\nsetup(name='{package_name}', version='0.1.0')\n"
    )
    record = PackageRecord(
        package_name=package_name.strip(),
        version="0.1.0",
        label="unknown",
        source="mcp_agent_request",
        summary="MCP pre-install audit request",
        setup_code=clean_code,
        age_days=float(age_days),
        release_count=int(release_count),
        code_size_bytes=len(clean_code),
        shannon_entropy=calculate_shannon_entropy(clean_code),
    )
    with contextlib.redirect_stdout(sys.stderr):
        report = judge.audit_package(record, check_live_registry=True)
    return report.model_dump_json(indent=2)


@mcp.tool()
def audit_requirements_manifest(requirements_txt: str, send_mobile_alert: bool = True) -> str:
    """
    Audits an entire requirements.txt manifest, strips malicious or slopsquatted
    dependencies, triggers a Pushover mobile alert on BLOCK verdicts, and returns
    a sanitized requirements.safe.txt lockfile.
    """
    _, manifest_agent = _get_engines()
    with contextlib.redirect_stdout(sys.stderr):
        summary = manifest_agent.audit_manifest(
            requirements_text=requirements_txt,
            send_pushover=send_mobile_alert,
        )
    return json.dumps(
        {
            "total_packages": summary.total_packages,
            "allowed": summary.allowed_count,
            "warned": summary.warn_count,
            "blocked": summary.blocked_count,
            "pushover_status": summary.pushover_status,
            "safe_lockfile": summary.safe_lockfile_content,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)