import itertools
import os
from pathlib import Path
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from D:\projects\llm_engineering\.env
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
ALLOWLIST_PATH = Path(__file__).resolve().parents[2] / "allowlist.txt"
load_dotenv(dotenv_path=ENV_PATH)

T = TypeVar("T", bound=BaseModel)

# Locked model identifiers
GEMINI_MODEL = "gemini-3.1-flash-lite"
OPENROUTER_MODEL = "openrouter/free"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_private_allowlist() -> set[str]:
    """Loads private/internal package names from allowlist.txt and ALLOWED_PRIVATE_PACKAGES env."""
    allowed: set[str] = set()
    env_pkgs = os.getenv("ALLOWED_PRIVATE_PACKAGES", "").strip()
    if env_pkgs:
        for item in env_pkgs.split(","):
            cleaned = item.strip().lower().replace("_", "-")
            if cleaned:
                allowed.add(cleaned)

    if ALLOWLIST_PATH.exists():
        for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                allowed.add(stripped.lower().replace("_", "-"))
    return allowed


class ResilientLLMRouter:
    """
    Rotates across GOOGLE_API_KEY .. GOOGLE_API_KEY_5 using gemini-3.1-flash-lite,
    and automatically falls back to openrouter/free if all Gemini keys hit rate limits.
    """

    def __init__(self) -> None:
        key_names = [
            "GOOGLE_API_KEY",
            "GOOGLE_API_KEY_2",
            "GOOGLE_API_KEY_3",
            "GOOGLE_API_KEY_4",
            "GOOGLE_API_KEY_5",
        ]
        self.gemini_keys: list[str] = [
            os.getenv(k, "").strip() for k in key_names if os.getenv(k, "").strip()
        ]
        self._key_cycle = itertools.cycle(self.gemini_keys) if self.gemini_keys else None
        self.openrouter_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()

        # Cloud & Alert credentials
        self.hf_token: str = os.getenv("HF_TOKEN", "").strip()
        self.modal_token_id: str = os.getenv("MODAL_TOKEN_ID", "").strip()
        self.modal_token_secret: str = os.getenv("MODAL_TOKEN_SECRET", "").strip()
        self.pushover_user: str = os.getenv("PUSHOVER_USER", "").strip()
        self.pushover_token: str = os.getenv("PUSHOVER_TOKEN", "").strip()

    def _next_gemini_client(self) -> OpenAI:
        if not self._key_cycle:
            raise RuntimeError("No GOOGLE_API_KEYs found in .env")
        api_key = next(self._key_cycle)
        return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    def _openrouter_client(self) -> OpenAI:
        if not self.openrouter_key:
            raise RuntimeError("No OPENROUTER_API_KEY found in .env")
        return OpenAI(api_key=self.openrouter_key, base_url=OPENROUTER_BASE_URL)

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        temperature: float = 0.1,
    ) -> tuple[T, str]:
        """
        Executes a schema-validated LLM call.
        Returns (parsed_pydantic_object, model_used_label).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None

        # 1. Try rotating across all available Gemini keys
        for _ in range(len(self.gemini_keys)):
            try:
                client = self._next_gemini_client()
                completion = client.beta.chat.completions.parse(
                    model=GEMINI_MODEL,
                    messages=messages,
                    response_format=response_schema,
                    temperature=temperature,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    return parsed, f"google/{GEMINI_MODEL}"
            except Exception as exc:
                last_error = exc
                continue

        # 2. Automatic failover to openrouter/free
        if self.openrouter_key:
            try:
                or_client = self._openrouter_client()
                completion = or_client.beta.chat.completions.parse(
                    model=OPENROUTER_MODEL,
                    messages=messages,
                    response_format=response_schema,
                    temperature=temperature,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    return parsed, OPENROUTER_MODEL
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> tuple[str, str]:
        """
        Executes a plain-text LLM call with Gemini key rotation and OpenRouter failover.
        Returns (response_text, model_used_label).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for _ in range(len(self.gemini_keys)):
            try:
                client = self._next_gemini_client()
                completion = client.chat.completions.create(
                    model=GEMINI_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                content = completion.choices[0].message.content
                if content:
                    return content.strip(), f"google/{GEMINI_MODEL}"
            except Exception as exc:
                last_error = exc
                continue

        if self.openrouter_key:
            try:
                or_client = self._openrouter_client()
                completion = or_client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                content = completion.choices[0].message.content
                if content:
                    return content.strip(), OPENROUTER_MODEL
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def send_pushover_alert(self, title: str, message: str, priority: int = 1) -> bool:
        """Sends an instant mobile alert via Pushover when malware/slopsquatting is blocked."""
        if not (self.pushover_user and self.pushover_token):
            return False
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.pushover_token,
                "user": self.pushover_user,
                "title": title,
                "message": message,
                "priority": priority,
            },
            timeout=10,
        )
        return resp.status_code == 200


# Singleton router instance imported across all DepShield modules
router = ResilientLLMRouter()