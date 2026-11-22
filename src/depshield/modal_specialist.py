import json
import os
import re
from pathlib import Path

import modal

from depshield.schemas import ModelSignal, PackageRecord

IS_REMOTE_MODAL = bool(os.getenv("MODAL_IMAGE_ID") or os.getenv("MODAL_TASK_ID"))

if not IS_REMOTE_MODAL:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ENV_PATH = PROJECT_ROOT.parent / ".env"
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
else:
    PROJECT_ROOT = Path("/root")
    PROCESSED_DIR = Path("/root/data/processed")

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
APP_NAME = "depshield-qlora-specialist"
VOLUME_NAME = "depshield-qlora-vol"
ADAPTER_DIR = "/vol/depshield_adapter"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Bundles depshield package source so modal deploy includes depshield.schemas
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        "python-dotenv==1.0.1",
        "pydantic>=2.9.0",
    )
    .add_local_python_source("depshield")
)


def format_code_only_prompt(setup_code: str) -> str:
    """
    Formats raw Python code WITHOUT metadata (age_days / release_count) so the
    QLoRA specialist learns purely from code syntax & behavior (no shortcut learning).
    """
    return (
        "### System:\n"
        "You are a Python supply-chain malware classifier. Analyze the package source code below "
        "and respond strictly with JSON: {\"label\": \"malicious\" or \"benign\", \"risk\": 0.00-1.00, \"reason\": \"...\"}\n\n"
        f"### Source Code:\n```python\n{setup_code[:2200]}\n```\n\n"
        "### Verdict:\n"
    )


@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=900,
    retries=0,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN", "")})],
)
def train_qlora_remote(train_records_json: list[str]) -> str:
    """
    Fine-tunes the 3B model using 4-bit QLoRA (NF4 + LoRA r=16) on Modal T4 GPU
    and saves adapter weights to the persistent Modal Volume (/vol/depshield_adapter).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    hf_token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    formatted_rows = []
    for raw_line in train_records_json:
        rec = json.loads(raw_line)
        prompt = format_code_only_prompt(rec["setup_code"])
        is_mal = rec["label"] == "malicious"
        target_json = json.dumps(
            {
                "label": "malicious" if is_mal else "benign",
                "risk": 0.96 if is_mal else 0.04,
                "reason": (
                    "Malicious supply-chain payload or unauthorized execution pattern detected."
                    if is_mal
                    else "Standard legitimate Python package structure."
                ),
            }
        )
        full_text = prompt + target_json + tokenizer.eos_token
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=768,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        formatted_rows.append(tokenized)

    ds = Dataset.from_list(formatted_rows)

    training_args = TrainingArguments(
        output_dir="/tmp/qlora_runs",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    volume.commit()
    return f"Successfully trained QLoRA adapter on {len(formatted_rows)} samples and committed to {ADAPTER_DIR}."


@app.cls(
    image=gpu_image,
    gpu="T4",
    scaledown_window=60,
    retries=0,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.getenv("HF_TOKEN", "")})],
)
class DepShieldQLoRAService:
    """Serverless Modal T4 GPU inference container serving our fine-tuned QLoRA specialist."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        volume.reload()
        hf_token = os.environ.get("HF_TOKEN") or None
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )
        if os.path.exists(ADAPTER_DIR):
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        else:
            self.model = base_model
        self.model.eval()

    @modal.method()
    def classify_batch(self, code_snippets: list[str]) -> list[dict]:
        import torch

        outputs: list[dict] = []
        for code in code_snippets:
            prompt = format_code_only_prompt(code)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=90,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1] :]
            raw_reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            parsed = {"label": "benign", "risk": 0.10, "reason": raw_reply[:200]}
            match = re.search(r"\{.*?\}", raw_reply, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    label = str(data.get("label", "benign")).lower()
                    risk = float(data.get("risk", 0.90 if "mal" in label else 0.10))
                    parsed = {
                        "label": "malicious" if "mal" in label else "benign",
                        "risk": round(min(1.0, max(0.0, risk)), 4),
                        "reason": str(data.get("reason", "QLoRA specialist verdict"))[:220],
                    }
                except Exception:
                    if "malicious" in raw_reply.lower():
                        parsed = {"label": "malicious", "risk": 0.88, "reason": raw_reply[:200]}
            elif "malicious" in raw_reply.lower():
                parsed = {"label": "malicious", "risk": 0.88, "reason": raw_reply[:200]}

            outputs.append(parsed)
        return outputs


class ModalQLoRAPredictor:
    """Local client connecting DepShield AI to our serverless Modal T4 QLoRA service."""

    def __init__(self) -> None:
        self.service_cls = modal.Cls.from_name(APP_NAME, "DepShieldQLoRAService")

    def predict_signal(self, record: PackageRecord) -> ModelSignal:
        res = self.service_cls().classify_batch.remote([record.setup_code])[0]
        prob = float(res["risk"])
        confidence = round(min(0.99, abs(prob - 0.5) * 2.0), 4)
        return ModelSignal(
            model_name="modal_qlora_specialist",
            risk_probability=prob,
            confidence=confidence,
            attack_category="obfuscated_dropper" if prob >= 0.5 else "benign",
            reasoning=f"[Modal T4 QLoRA] {res['reason']}",
        )


def run_training_and_benchmark(threshold: float = 0.50) -> dict[str, float]:
    """
    1. Trains QLoRA adapter on Modal T4 GPU using train.jsonl (210 real scripts).
    2. Invokes serverless DepShieldQLoRAService on test.jsonl (46 real scripts).
    3. Logs Row 4 of our Ablation Benchmark.
    """
    train_lines = [
        line
        for line in (PROCESSED_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_records = [
        PackageRecord.model_validate_json(line)
        for line in (PROCESSED_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with modal.enable_output():
        with app.run():
            print("Launching QLoRA fine-tuning on Modal NVIDIA T4 GPU...")
            train_msg = train_qlora_remote.remote(train_lines)
            print(train_msg)

            print(f"\nRunning serverless T4 QLoRA inference on {len(test_records)} held-out test scripts...")
            service = DepShieldQLoRAService()
            batch_results = service.classify_batch.remote([r.setup_code for r in test_records])

    tp = fp = tn = fn = 0
    for rec, res in zip(test_records, batch_results):
        pred_mal = float(res["risk"]) >= threshold
        actual_mal = rec.label == "malicious"
        if pred_mal and actual_mal:
            tp += 1
        elif pred_mal and not actual_mal:
            fp += 1
        elif not pred_mal and not actual_mal:
            tn += 1
        else:
            fn += 1

    total = len(test_records)
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "stage": "Stage 2C: Modal T4 QLoRA Specialist (Code-Only)",
        "accuracy": round(accuracy * 100, 2),
        "recall_detection_rate": round(recall * 100, 2),
        "false_positive_rate": round(fpr * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }

    print("\n=== STAGE 2C MODAL QLoRA BENCHMARK (test.jsonl: 46 real packages, Code-Only) ===")
    print(f"TP={tp} | FN={fn} | TN={tn} | FP={fp}")
    print(f"Accuracy:            {metrics['accuracy']}%")
    print(f"Detection Rate (TP): {metrics['recall_detection_rate']}%")
    print(f"False-Positive Rate: {metrics['false_positive_rate']}%")
    print(f"F1-Score:            {metrics['f1_score']}%")
    return metrics


if __name__ == "__main__":
    run_training_and_benchmark()