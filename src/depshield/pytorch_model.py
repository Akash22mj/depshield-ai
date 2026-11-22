import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from depshield.schemas import ModelSignal, PackageRecord, StaticScanResult
from depshield.static_scanner import scan_package_static

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "tabular_risk_net.pt"

FEATURE_DIM = 10


def extract_tabular_features(
    record: PackageRecord,
    static_res: StaticScanResult | None = None,
) -> np.ndarray:
    """
    Converts package metadata and Stage 1 AST signals into a 10-dimensional numeric
    feature vector completely immune to text-based prompt injection.
    """
    if static_res is None:
        static_res = scan_package_static(record, check_live_registry=False)

    age_log = math.log1p(max(0.0, record.age_days))
    releases_log = math.log1p(max(0, record.release_count))
    size_log = math.log1p(max(0, record.code_size_bytes))
    entropy = float(record.shannon_entropy)
    num_susp_imports = float(len(static_res.suspicious_imports))
    num_susp_calls = float(len(static_res.suspicious_calls))
    install_hook = 1.0 if static_res.has_install_hook else 0.0
    is_typosquat = 1.0 if static_res.typosquat_target else 0.0
    static_score = float(static_res.static_risk_score)
    entropy_density = entropy / max(0.25, record.code_size_bytes / 1024.0)

    return np.array(
        [
            age_log,
            releases_log,
            size_log,
            entropy,
            num_susp_imports,
            num_susp_calls,
            install_hook,
            is_typosquat,
            static_score,
            min(25.0, entropy_density),
        ],
        dtype=np.float32,
    )


class TabularRiskNet(nn.Module):
    """Feed-forward neural network predicting supply-chain malware probability."""

    def __init__(self, in_features: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_split_tensors(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / split_name
    records = [
        PackageRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    X = np.vstack([extract_tabular_features(r) for r in records])
    y = np.array([1.0 if r.label == "malicious" else 0.0 for r in records], dtype=np.float32)
    return X, y


def train_pytorch_model(epochs: int = 80, batch_size: int = 32, lr: float = 2e-3) -> None:
    """
    Trains TabularRiskNet on train.jsonl, selects best weights using val.jsonl,
    and saves model + StandardScaler normalization parameters to models/tabular_risk_net.pt.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    X_train_raw, y_train = _load_split_tensors("train.jsonl")
    X_val_raw, y_val = _load_split_tensors("val.jsonl")

    # Fit StandardScaler strictly on train.jsonl (zero validation/test leakage)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-6

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TabularRiskNet(in_features=FEATURE_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_loss = float("inf")
    best_state = None

    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = float((val_preds == y_val_t).float().mean().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    torch.save(
        {
            "state_dict": best_state,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved best PyTorch checkpoint (best val_loss={best_val_loss:.4f}) to {CHECKPOINT_PATH}.")


class PyTorchTabularPredictor:
    """Inference wrapper loading trained TabularRiskNet + normalization stats."""

    def __init__(self) -> None:
        if not CHECKPOINT_PATH.exists():
            train_pytorch_model()
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.mean = np.array(ckpt["mean"], dtype=np.float32)
        self.std = np.array(ckpt["std"], dtype=np.float32)
        self.model = TabularRiskNet(in_features=FEATURE_DIM)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_signal(
        self,
        record: PackageRecord,
        static_res: StaticScanResult | None = None,
    ) -> ModelSignal:
        feats = extract_tabular_features(record, static_res)
        norm_feats = (feats - self.mean) / self.std
        with torch.no_grad():
            logit = self.model(torch.tensor(norm_feats).unsqueeze(0))
            prob = round(float(torch.sigmoid(logit).item()), 4)

        confidence = round(min(0.99, abs(prob - 0.5) * 2.0), 4)
        category = "suspicious_unknown" if prob >= 0.5 else "benign"
        reasoning = (
            f"PyTorch neural network evaluated 10 numeric signals "
            f"(age={record.age_days}d, releases={record.release_count}, "
            f"entropy={record.shannon_entropy}) -> risk={prob:.2f}."
        )
        return ModelSignal(
            model_name="pytorch_tabular_net",
            risk_probability=prob,
            confidence=confidence,
            attack_category=category,
            reasoning=reasoning,
        )


def evaluate_pytorch_on_test_set(threshold: float = 0.50) -> dict[str, float]:
    """Trains TabularRiskNet and logs Row 3 of our Ablation Benchmark on test.jsonl."""
    train_pytorch_model()
    predictor = PyTorchTabularPredictor()

    test_path = PROCESSED_DIR / "test.jsonl"
    records = [
        PackageRecord.model_validate_json(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tp = fp = tn = fn = 0
    for rec in records:
        sig = predictor.predict_signal(rec)
        pred_mal = sig.risk_probability >= threshold
        actual_mal = rec.label == "malicious"

        if pred_mal and actual_mal:
            tp += 1
        elif pred_mal and not actual_mal:
            fp += 1
        elif not pred_mal and not actual_mal:
            tn += 1
        else:
            fn += 1

    total = len(records)
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "stage": "Stage 2B: Local PyTorch TabularRiskNet",
        "accuracy": round(accuracy * 100, 2),
        "recall_detection_rate": round(recall * 100, 2),
        "false_positive_rate": round(fpr * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }

    print("\n=== STAGE 2B PYTORCH BENCHMARK (test.jsonl: 46 real packages) ===")
    print(f"TP={tp} | FN={fn} | TN={tn} | FP={fp}")
    print(f"Accuracy:            {metrics['accuracy']}%")
    print(f"Detection Rate (TP): {metrics['recall_detection_rate']}%")
    print(f"False-Positive Rate: {metrics['false_positive_rate']}%")
    print(f"F1-Score:            {metrics['f1_score']}%")
    return metrics


if __name__ == "__main__":
    evaluate_pytorch_on_test_set()