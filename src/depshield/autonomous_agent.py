import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from depshield.dataset_builder import calculate_shannon_entropy
from depshield.ensemble_judge import EnsembleAuditReport, EnsembleJudge
from depshield.schemas import PackageRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


class ManifestAuditSummary(BaseModel):
    """Structured output produced by the Autonomous Manifest Planning Agent."""

    total_packages: int
    allowed_count: int
    warn_count: int
    blocked_count: int
    reports: list[EnsembleAuditReport] = Field(default_factory=list)
    safe_lockfile_content: str
    pushover_sent: bool = False
    pushover_status: str = "Not triggered"


class PushoverAlertAgent:
    """
    Messaging Agent (Ed Donner Core Track Wk 8 + Agentic Track):
    Dispatches real-time mobile notifications via Pushover when high-risk threats are blocked.
    """

    def __init__(self) -> None:
        self.user_key = os.getenv("PUSHOVER_USER", "").strip()
        self.api_token = os.getenv("PUSHOVER_TOKEN", "").strip()

    def send_security_alert(self, title: str, message: str, priority: int = 1) -> tuple[bool, str]:
        if not self.user_key or not self.api_token:
            return False, "Skipped (PUSHOVER_USER or PUSHOVER_TOKEN missing in .env)"

        try:
            resp = httpx.post(
                PUSHOVER_API_URL,
                data={
                    "token": self.api_token,
                    "user": self.user_key,
                    "title": title[:100],
                    "message": message[:1000],
                    "priority": priority,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                return True, "Delivered mobile alert via Pushover API"
            return False, f"Pushover HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as exc:
            return False, f"Pushover error: {str(exc)[:120]}"


class AutonomousSupplyChainAgent:
    """
    Autonomous Planning & Remediation Agent:
    1. Parses requirements.txt lines into structured package targets.
    2. Queries live PyPI metadata (age, release count, summary) for each dependency.
    3. Runs the 4-Stage Orthogonal Ensemble Judge across all packages.
    4. Plans remediation: strips BLOCKED packages, flags WARN items, and outputs a pinned safe manifest.
    5. Triggers PushoverAlertAgent if any supply-chain threat is blocked.
    """

    def __init__(self, use_modal: bool = False) -> None:
        self.judge = EnsembleJudge(use_modal=use_modal)
        self.alerter = PushoverAlertAgent()

    @staticmethod
    def parse_requirements(raw_text: str) -> list[tuple[str, str]]:
        parsed: list[tuple[str, str]] = []
        for line in raw_text.splitlines():
            clean = line.split("#")[0].strip()
            if not clean or clean.startswith("-"):
                continue
            match = re.match(r"^([A-Za-z0-9_.\-]+)\s*([=<>!~]=?\s*[A-Za-z0-9_.\-]+)?", clean)
            if match:
                pkg = match.group(1).strip()
                spec = (match.group(2) or "").strip()
                parsed.append((pkg, spec))
        return parsed

    @staticmethod
    def fetch_live_package_record(pkg_name: str, version_spec: str = "") -> PackageRecord:
        """Fetches real PyPI registry metadata or constructs a suspicious zero-day record if absent."""
        url = f"https://pypi.org/pypi/{pkg_name}/json"
        try:
            resp = httpx.get(url, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("info", {})
                releases = data.get("releases", {})
                version = info.get("version", "0.1.0")
                summary = (info.get("summary") or "")[:400]
                release_count = max(1, len(releases))

                # Estimate package age from earliest release upload timestamp
                earliest_year = 2026
                for rel_files in releases.values():
                    for f in rel_files:
                        ts = f.get("upload_time", "")
                        if len(ts) >= 4 and ts[:4].isdigit():
                            earliest_year = min(earliest_year, int(ts[:4]))
                age_days = max(1.0, float((2026 - earliest_year) * 365 + 30))

                synthetic_setup = (
                    f"from setuptools import setup\n"
                    f"setup(name='{pkg_name}', version='{version}', description={summary!r})\n"
                )
                return PackageRecord(
                    package_name=pkg_name,
                    version=version,
                    label="unknown",
                    source="pypi_live",
                    summary=summary,
                    setup_code=synthetic_setup,
                    age_days=age_days,
                    release_count=release_count,
                    code_size_bytes=len(synthetic_setup),
                    shannon_entropy=calculate_shannon_entropy(synthetic_setup),
                )
        except Exception:
            pass

        # Package missing on PyPI or unreachable -> treat as high-risk unverified zero-day
        fallback_code = f"from setuptools import setup\nsetup(name='{pkg_name}', version='0.1.0')\n"
        return PackageRecord(
            package_name=pkg_name,
            version=version_spec.lstrip("=<>~") or "0.1.0",
            label="unknown",
            source="unverified_manifest",
            summary="Unverified package missing from PyPI registry",
            setup_code=fallback_code,
            age_days=0.5,
            release_count=1,
            code_size_bytes=len(fallback_code),
            shannon_entropy=calculate_shannon_entropy(fallback_code),
        )

    def audit_manifest(
        self,
        requirements_text: str,
        send_pushover: bool = True,
    ) -> ManifestAuditSummary:
        targets = self.parse_requirements(requirements_text)
        reports: list[EnsembleAuditReport] = []
        safe_lines: list[str] = [
            "# Generated by DepShield AI Autonomous Supply-Chain Agent",
        ]

        allowed = warned = blocked = 0
        blocked_names: list[str] = []

        for pkg_name, spec in targets:
            rec = self.fetch_live_package_record(pkg_name, spec)
            report = self.judge.audit_package(rec, check_live_registry=True)
            reports.append(report)

            if report.verdict == "BLOCK":
                blocked += 1
                blocked_names.append(f"{pkg_name} ({report.final_risk_score:.0%})")
                safe_lines.append(f"# [BLOCKED BY DEPSHIELD AI - Risk {report.final_risk_score:.2%}] {pkg_name}{spec}")
            elif report.verdict == "WARN":
                warned += 1
                safe_lines.append(f"# [WARN - REVIEW] {pkg_name}=={rec.version}")
            else:
                allowed += 1
                pinned = spec if spec else f"=={rec.version}"
                safe_lines.append(f"{pkg_name}{pinned}")

        push_sent = False
        push_status = "No threats blocked"
        if blocked > 0 and send_pushover:
            alert_msg = (
                f"DepShield AI blocked {blocked} malicious/slopsquatted package(s) in manifest: "
                f"{', '.join(blocked_names)}."
            )
            push_sent, push_status = self.alerter.send_security_alert(
                title="DepShield AI: Supply-Chain Threat Blocked",
                message=alert_msg,
            )

        return ManifestAuditSummary(
            total_packages=len(targets),
            allowed_count=allowed,
            warn_count=warned,
            blocked_count=blocked,
            reports=reports,
            safe_lockfile_content="\n".join(safe_lines) + "\n",
            pushover_sent=push_sent,
            pushover_status=push_status,
        )


if __name__ == "__main__":
    sample_manifest = (
        "requests==2.32.3\n"
        "fastapi>=0.115.0\n"
        "reqeusts-auth-utils==0.1.0\n"
    )
    agent = AutonomousSupplyChainAgent(use_modal=False)
    res = agent.audit_manifest(sample_manifest, send_pushover=True)
    print(f"Scanned {res.total_packages} packages -> ALLOW={res.allowed_count}, WARN={res.warn_count}, BLOCK={res.blocked_count}")
    print(f"Pushover Status: {res.pushover_status}")
    print("\n--- Generated Safe Lockfile ---")
    print(res.safe_lockfile_content)