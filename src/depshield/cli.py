import argparse
import os
import subprocess
import sys
from pathlib import Path

from depshield.autonomous_agent import AutonomousSupplyChainAgent
from depshield.ensemble_judge import EnsembleJudge
from depshield.schemas import PackageRecord


def cmd_install(packages: list[str], use_modal: bool = False, yes: bool = False) -> int:
    judge = EnsembleJudge(use_modal=use_modal)
    approved: list[str] = []

    for raw_spec in packages:
        pkg_name = raw_spec.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
        print(f"[DepShield] Auditing '{pkg_name}' before pip install...")
        rec = PackageRecord(
            package_name=pkg_name,
            version="latest",
            label="unknown",
            source="cli_wrapper",
            summary=f"CLI install audit for {raw_spec}",
            setup_code=f"from setuptools import setup\nsetup(name='{pkg_name}', version='0.1.0')\n",
            age_days=1.0,
            release_count=1,
            code_size_bytes=72,
            shannon_entropy=4.1,
        )
        report = judge.audit_package(rec, check_live_registry=True)
        print(f"  -> {report.executive_summary}")

        if report.verdict == "BLOCK":
            print(f"[BLOCKED] Aborting installation of '{raw_spec}' (Risk: {report.final_risk_score:.2%}).")
            return 1

        if report.verdict == "WARN" and not yes:
            choice = input(
                f"[WARN] '{raw_spec}' flagged with {report.final_risk_score:.2%} risk. Proceed anyway? [y/N]: "
            ).strip().lower()
            if choice != "y":
                print(f"[ABORTED] Skipped '{raw_spec}'.")
                return 1

        approved.append(raw_spec)

    if not approved:
        return 0

    print(f"[DepShield] All packages verified. Running: {sys.executable} -m pip install {' '.join(approved)}")
    completed = subprocess.run([sys.executable, "-m", "pip", "install", *approved], check=False)
    return completed.returncode


def cmd_audit_manifest(manifest_path: str, send_pushover: bool = False) -> int:
    path = Path(manifest_path)
    if not path.exists():
        print(f"[ERROR] Manifest file not found: {path}")
        return 1

    agent = AutonomousSupplyChainAgent(use_modal=False)
    summary = agent.audit_manifest(
        requirements_text=path.read_text(encoding="utf-8"),
        send_pushover=send_pushover,
    )
    print(
        f"[DepShield] Scanned {summary.total_packages} packages -> "
        f"ALLOW: {summary.allowed_count} | BLOCK: {summary.blocked_count}"
    )
    for rep in summary.reports:
        print(f"  - {rep.package_name}: {rep.verdict} ({rep.final_risk_score:.2%}) — {rep.executive_summary}")

    if summary.blocked_count > 0:
        safe_path = path.with_name("requirements.safe.txt")
        safe_path.write_text(summary.safe_lockfile_content, encoding="utf-8")
        print(f"[BLOCKED] Wrote sanitized dependencies to {safe_path}. Failing check.")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="depshield", description="DepShield AI Supply-Chain Firewall CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inst = sub.add_parser("install", help="Audit packages with DepShield AI before running pip install")
    p_inst.add_argument("packages", nargs="+", help="Package specifiers (e.g., requests fastapi>=0.115.0)")
    p_inst.add_argument("--modal", action="store_true", help="Enable Modal T4 QLoRA stage")
    p_inst.add_argument("--private-index", default="", help="Optional private PEP-503 index URL (sets PRIVATE_INDEX_URL)")
    p_inst.add_argument("-y", "--yes", action="store_true", help="Auto-approve WARN verdicts")

    p_aud = sub.add_parser("audit-manifest", help="Audit a requirements.txt file (for CI/CD & PR hooks)")
    p_aud.add_argument("manifest", nargs="?", default="requirements.txt", help="Path to requirements.txt")
    p_aud.add_argument("--pushover", action="store_true", help="Send Pushover mobile alert on BLOCK")

    args = parser.parse_args()
    if args.command == "install":
        if args.private_index:
            os.environ["PRIVATE_INDEX_URL"] = args.private_index
        raise SystemExit(cmd_install(args.packages, use_modal=args.modal, yes=args.yes))
    if args.command == "audit-manifest":
        raise SystemExit(cmd_audit_manifest(args.manifest, send_pushover=args.pushover))


if __name__ == "__main__":
    main()