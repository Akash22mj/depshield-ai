import ast
import json
import os
from pathlib import Path

import requests

from depshield.schemas import PackageRecord, StaticScanResult

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

PROTECTED_POPULAR_PACKAGES = [
    "requests", "fastapi", "pydantic", "httpx", "uvicorn", "numpy", "pandas",
    "scipy", "scikit-learn", "torch", "tensorflow", "transformers", "openai",
    "anthropic", "langchain", "langgraph", "chromadb", "gradio", "streamlit",
    "flask", "django", "sqlalchemy", "boto3", "paramiko", "cryptography",
    "pyyaml", "pillow", "beautifulsoup4", "selenium", "playwright", "pytest",
    "celery", "redis", "pymongo", "psycopg2", "aiohttp", "websockets",
]

DANGEROUS_MODULES = {
    "subprocess", "socket", "ctypes", "winreg", "ftplib", "smtplib",
    "telnetlib", "marshal", "pickle", "pty", "commands", "webbrowser",
}

NETWORK_OR_ENCODING_MODULES = {
    "urllib", "urllib.request", "requests", "httpx", "aiohttp",
    "base64", "codecs", "binascii", "zlib",
}

DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "system", "popen",
    "Popen", "call", "check_output", "run", "b64decode", "urlsafe_b64decode",
    "urlopen", "urlretrieve", "getattr",
}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Computes edit distance between two package names to catch typosquatting."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_typosquat_target(package_name: str) -> str | None:
    """Returns the mimicked popular package if edit distance is 1 or 2."""
    clean = package_name.lower().replace("_", "-").strip()
    if clean in PROTECTED_POPULAR_PACKAGES:
        return None
    for popular in PROTECTED_POPULAR_PACKAGES:
        if abs(len(clean) - len(popular)) <= 2 and len(popular) >= 5:
            dist = _levenshtein_distance(clean, popular)
            if 1 <= dist <= 2:
                return popular
    return None


def check_pypi_exists(package_name: str, private_index_url: str | None = None) -> bool:
    """Checks live PyPI (and optional private PEP-503 index) to catch slopsquatting."""
    headers = {"User-Agent": "DepShield-AI-Research/1.0"}
    try:
        resp = requests.get(
            f"https://pypi.org/pypi/{package_name}/json",
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 200:
            return True
    except Exception:
        return True

    idx_url = (private_index_url or os.getenv("PRIVATE_INDEX_URL", "")).strip().rstrip("/")
    if idx_url:
        try:
            norm = package_name.lower().replace("_", "-")
            p_resp = requests.get(f"{idx_url}/{norm}/", headers=headers, timeout=8)
            if p_resp.status_code == 200:
                return True
        except Exception:
            pass
    return False


def scan_package_static(
    record: PackageRecord,
    check_live_registry: bool = False,
    private_index_url: str | None = None,
) -> StaticScanResult:
    """
    Stage 1 Deterministic Baseline:
    Parses Python AST safely without execution, checks typosquatting edit distance,
    inspects Shannon entropy, and optionally verifies live PyPI / private registry existence.
    """
    reasons: list[str] = []
    risk = 0.05

    exists_on_pypi = True
    if check_live_registry:
        exists_on_pypi = check_pypi_exists(record.package_name, private_index_url=private_index_url)
        if not exists_on_pypi:
            risk += 0.45
            reasons.append(f"Package '{record.package_name}' does not exist on PyPI (slopsquatting risk)")

    typo_target = find_typosquat_target(record.package_name)
    if typo_target:
        risk += 0.35
        reasons.append(f"Name closely mimics popular package '{typo_target}' (typosquatting)")

    suspicious_imports: set[str] = set()
    suspicious_calls: set[str] = set()
    has_install_hook = False
    long_encoded_blobs = 0

    try:
        tree = ast.parse(record.setup_code.replace("\x00", ""))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in DANGEROUS_MODULES or alias.name in NETWORK_OR_ENCODING_MODULES:
                        suspicious_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in DANGEROUS_MODULES or node.module in NETWORK_OR_ENCODING_MODULES:
                    suspicious_imports.add(node.module)
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in DANGEROUS_CALLS:
                    suspicious_calls.add(func_name)
                if func_name == "setup":
                    for kw in node.keywords:
                        if kw.arg == "cmdclass":
                            has_install_hook = True
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if len(val) >= 80 and " " not in val and "\n" not in val:
                    long_encoded_blobs += 1
    except SyntaxError:
        risk += 0.25
        reasons.append("AST syntax parse failed (possible obfuscated/non-Python payload)")

    if has_install_hook:
        risk += 0.25
        reasons.append("Overrides setup.py 'cmdclass' install hook (executes code on pip install)")
    if suspicious_imports:
        risk += min(0.30, 0.12 * len(suspicious_imports))
        reasons.append(f"Suspicious imports detected: {sorted(suspicious_imports)}")
    if suspicious_calls:
        risk += min(0.40, 0.15 * len(suspicious_calls))
        reasons.append(f"Dangerous runtime calls detected: {sorted(suspicious_calls)}")
    if long_encoded_blobs > 0:
        risk += 0.25
        reasons.append(f"Found {long_encoded_blobs} long unbroken string literal(s) (possible base64/hex payload)")
    if record.shannon_entropy > 5.2:
        risk += 0.20
        reasons.append(f"High Shannon code entropy ({record.shannon_entropy:.2f} > 5.20)")

    final_score = round(min(1.0, max(0.0, risk)), 4)

    return StaticScanResult(
        package_name=record.package_name,
        exists_on_pypi=exists_on_pypi,
        typosquat_target=typo_target,
        suspicious_imports=sorted(suspicious_imports),
        suspicious_calls=sorted(suspicious_calls),
        has_install_hook=has_install_hook,
        static_risk_score=final_score,
        reasons=reasons,
    )