"""
lie_detector.py

Trains a lie detector probe on saved layer-wise hidden states.
Uses the critical layers identified by the control experiment:
  - Llama: layers 14-15
  - Qwen:  layers 1-4 (highest AUC in layerwise analysis)

Evaluates on held-out questions NOT seen during training.
Outputs:
  - Trained probe (saved as .pkl)
  - Precision, Recall, F1, AUC on held-out set
  - Threshold calibration curve (for different industry deployments)
  - Per-question detection results

Usage:
  python experiments/lie_detector.py \
      --hiddens results/layerwise/qwen \
      --model_name qwen \
      --critical_layers 1 2 3 4 \
      --out results/detector/qwen

  python experiments/lie_detector.py \
      --hiddens results/layerwise/llama \
      --model_name llama \
      --critical_layers 14 15 \
      --out results/detector/llama
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
    classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats


# ── Geometry utils ────────────────────────────────────────────────────────────

def l2_normalize(h: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(h, axis=-1, keepdims=True)
    return h / np.maximum(norms, 1e-8)

def to_poincare(h: np.ndarray) -> np.ndarray:
    return np.tanh(0.5) * l2_normalize(h)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    hidden_states: np.ndarray,
    critical_layers: list,
    use_poincare: bool = False,
) -> np.ndarray:
    """
    Extract features from critical layers only.
    hidden_states: (N, n_layers, hidden_dim)
    Returns: (N, len(critical_layers) * hidden_dim) — concatenated layer features
    """
    features = []
    for l in critical_layers:
        h = hidden_states[:, l, :]          # (N, hidden_dim)
        if use_poincare:
            h = to_poincare(h)
        else:
            h = l2_normalize(h)
        features.append(h)

    return np.concatenate(features, axis=1)  # (N, layers * hidden_dim)


def reduce_features(X: np.ndarray, n_components: int = 100) -> tuple:
    """
    SVD dimensionality reduction.
    Returns (X_reduced, Vt) — Vt needed to transform new samples.
    """
    X_c = X - X.mean(0)
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(X_c, full_matrices=False)
        Vt = Vt[:n_components]
        return X_c @ Vt.T, Vt, X.mean(0)
    except Exception:
        return X_c[:, :n_components], None, X.mean(0)


# ── Probe training ────────────────────────────────────────────────────────────

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
) -> tuple:
    """Train logistic regression probe. Returns (clf, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


# ── Cross validation ──────────────────────────────────────────────────────────

def cross_validate_probe(
    truth_hiddens: np.ndarray,
    lie_hiddens: np.ndarray,
    critical_layers: list,
    use_poincare: bool = False,
    n_splits: int = 5,
) -> dict:
    """
    5-fold stratified CV on full dataset.
    Returns dict of metrics.
    """
    X_truth = extract_features(truth_hiddens, critical_layers, use_poincare)
    X_lie   = extract_features(lie_hiddens,   critical_layers, use_poincare)
    X = np.concatenate([X_truth, X_lie], axis=0)
    y = np.array([0]*len(X_truth) + [1]*len(X_lie))

    # Dimensionality reduction
    X_c   = X - X.mean(0)
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(X_c, full_matrices=False)
        X_proj = X_c @ Vt[:100].T
    except Exception:
        X_proj = X_c[:, :100]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs, precs, recs = [], [], [], [], []

    for tr, te in skf.split(X_proj, y):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_proj[tr])
        Xte = sc.transform(X_proj[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(Xtr, y[tr])

        preds = clf.predict(Xte)
        probs = clf.predict_proba(Xte)[:, 1]

        accs.append((preds == y[te]).mean())
        f1s.append(f1_score(y[te], preds, zero_division=0))
        precs.append(precision_score(y[te], preds, zero_division=0))
        recs.append(recall_score(y[te], preds, zero_division=0))
        try:
            aucs.append(roc_auc_score(y[te], probs))
        except Exception:
            aucs.append(0.5)

    return {
        "accuracy":  float(np.mean(accs)),
        "f1":        float(np.mean(f1s)),
        "precision": float(np.mean(precs)),
        "recall":    float(np.mean(recs)),
        "auc":       float(np.mean(aucs)),
        "acc_std":   float(np.std(accs)),
        "auc_std":   float(np.std(aucs)),
    }


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> dict:
    """
    Find optimal thresholds for different industry risk tolerances.
    Lower threshold = more conservative = catches more lies but more false alarms.
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        fpr       = fp / (fp + tn + 1e-8)  # false positive rate
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "threshold": float(t),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "fpr":       float(fpr),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        })

    # Industry-specific recommendations
    industry_thresholds = {}

    # Healthcare: maximize recall (catch all lies), accept high FPR
    healthcare = max(results, key=lambda x: x["recall"] - 0.3*x["fpr"])
    industry_thresholds["healthcare"] = healthcare

    # Legal: balance precision and recall
    legal = max(results, key=lambda x: x["f1"])
    industry_thresholds["legal"] = legal

    # Finance: high precision (avoid false alarms)
    finance = max(results, key=lambda x: x["precision"] - 0.2*(1-x["recall"]))
    industry_thresholds["finance"] = finance

    # General: standard F1 optimal
    general = max(results, key=lambda x: x["f1"])
    industry_thresholds["general"] = general

    return {
        "all_thresholds": results,
        "industry": industry_thresholds,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hiddens",         type=str, required=True,
                   help="Dir with truth_all_layers.npy and lie_all_layers.npy")
    p.add_argument("--model_name",      type=str, required=True,
                   help="qwen or llama")
    p.add_argument("--critical_layers", type=int, nargs="+", required=True,
                   help="Which layers to use for detection")
    p.add_argument("--out",             type=str, required=True)
    p.add_argument("--test_split",      type=float, default=0.2,
                   help="Fraction of data for held-out test set")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load saved hidden states ──────────────────────────────────────────────
    hiddens_dir = Path(args.hiddens)
    print(f"Loading hidden states from {hiddens_dir}...")

    truth_all = np.load(hiddens_dir / "truth_all_layers.npy")
    lie_all   = np.load(hiddens_dir / "lie_all_layers.npy")

    n_pairs, n_layers, hidden_dim = truth_all.shape
    print(f"Loaded: {n_pairs} pairs × {n_layers} layers × {hidden_dim} dims")
    print(f"Critical layers: {args.critical_layers}")

    # ── Train/test split ──────────────────────────────────────────────────────
    n_test  = max(4, int(n_pairs * args.test_split))
    n_train = n_pairs - n_test

    # Use last n_test pairs as test set
    truth_train = truth_all[:n_train]
    lie_train   = lie_all[:n_train]
    truth_test  = truth_all[n_train:]
    lie_test    = lie_all[n_train:]

    print(f"Train: {n_train} pairs | Test: {n_test} pairs")

    # ── Cross-validation on train set ─────────────────────────────────────────
    print(f"\n── Cross-validation (train set, 5-fold) ─────────────────────")

    print("  Euclidean space:")
    cv_euc = cross_validate_probe(
        truth_train, lie_train, args.critical_layers, use_poincare=False
    )
    print(f"    AUC={cv_euc['auc']:.3f}±{cv_euc['auc_std']:.3f} | "
          f"F1={cv_euc['f1']:.3f} | "
          f"Prec={cv_euc['precision']:.3f} | "
          f"Rec={cv_euc['recall']:.3f}")

    print("  Poincaré space:")
    cv_hyp = cross_validate_probe(
        truth_train, lie_train, args.critical_layers, use_poincare=True
    )
    print(f"    AUC={cv_hyp['auc']:.3f}±{cv_hyp['auc_std']:.3f} | "
          f"F1={cv_hyp['f1']:.3f} | "
          f"Prec={cv_hyp['precision']:.3f} | "
          f"Rec={cv_hyp['recall']:.3f}")

    # ── Train final probe on ALL train data ───────────────────────────────────
    print(f"\n── Training final probe on full train set ───────────────────")

    # Extract features
    X_train_truth = extract_features(truth_train, args.critical_layers, False)
    X_train_lie   = extract_features(lie_train,   args.critical_layers, False)
    X_train = np.concatenate([X_train_truth, X_train_lie], axis=0)
    y_train = np.array([0]*len(X_train_truth) + [1]*len(X_train_lie))

    X_test_truth = extract_features(truth_test, args.critical_layers, False)
    X_test_lie   = extract_features(lie_test,   args.critical_layers, False)
    X_test = np.concatenate([X_test_truth, X_test_lie], axis=0)
    y_test = np.array([0]*len(X_test_truth) + [1]*len(X_test_lie))

    # Reduce dimensions
    X_train_c  = X_train - X_train.mean(0)
    mean_vec   = X_train.mean(0)
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(X_train_c, full_matrices=False)
        Vt = Vt[:100]
        X_train_r = X_train_c @ Vt.T
        X_test_r  = (X_test - mean_vec) @ Vt.T
    except Exception:
        Vt = None
        X_train_r = X_train_c[:, :100]
        X_test_r  = (X_test - mean_vec)[:, :100]

    # Train
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_r)
    X_test_sc  = scaler.transform(X_test_r)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train_sc, y_train)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print(f"\n── Held-out test set evaluation ─────────────────────────────")
    y_pred  = clf.predict(X_test_sc)
    y_probs = clf.predict_proba(X_test_sc)[:, 1]

    try:
        test_auc = roc_auc_score(y_test, y_probs)
    except Exception:
        test_auc = 0.5

    test_f1   = f1_score(y_test, y_pred, zero_division=0)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec  = recall_score(y_test, y_pred, zero_division=0)
    test_acc  = (y_pred == y_test).mean()

    print(f"  AUC:       {test_auc:.3f}")
    print(f"  F1:        {test_f1:.3f}")
    print(f"  Precision: {test_prec:.3f}")
    print(f"  Recall:    {test_rec:.3f}")
    print(f"  Accuracy:  {test_acc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Truth','Lie'])}")

    # Per-sample results
    print(f"  Per-sample predictions:")
    labels = ["Truth"]*len(X_test_truth) + ["Lie"]*len(X_test_lie)
    for i, (true_l, pred_l, prob) in enumerate(
        zip(labels, ["Truth" if p==0 else "Lie" for p in y_pred], y_probs)
    ):
        correct = "✅" if true_l == pred_l else "❌"
        print(f"    Sample {i+1}: True={true_l:5s} | "
              f"Pred={pred_l:5s} | P(lie)={prob:.3f} {correct}")

    # ── Threshold calibration ─────────────────────────────────────────────────
    print(f"\n── Industry threshold calibration ───────────────────────────")
    calib = calibrate_thresholds(y_test, y_probs)

    for industry, rec in calib["industry"].items():
        print(f"  {industry:12s}: "
              f"threshold={rec['threshold']:.2f} | "
              f"recall={rec['recall']:.3f} | "
              f"precision={rec['precision']:.3f} | "
              f"FPR={rec['fpr']:.3f}")

    # ── Save everything ───────────────────────────────────────────────────────
    probe_data = {
        "clf":            clf,
        "scaler":         scaler,
        "Vt":             Vt,
        "mean_vec":       mean_vec,
        "critical_layers": args.critical_layers,
        "model_name":     args.model_name,
    }
    with open(out_dir / "probe.pkl", "wb") as f:
        pickle.dump(probe_data, f)

    results = {
        "model":           args.model_name,
        "critical_layers": args.critical_layers,
        "cv_euclidean":    cv_euc,
        "cv_hyperbolic":   cv_hyp,
        "test": {
            "auc":       test_auc,
            "f1":        float(test_f1),
            "precision": float(test_prec),
            "recall":    float(test_rec),
            "accuracy":  float(test_acc),
        },
        "calibration": calib["industry"],
    }
    with open(out_dir / "detector_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nProbe saved → {out_dir}/probe.pkl")
    print(f"Results saved → {out_dir}/detector_results.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ROC curve
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_probs)
        axes[0].plot(fpr_arr, tpr_arr, color="#E91E63", linewidth=2,
                     label=f"Detector (AUC={test_auc:.3f})")
        axes[0].plot([0,1],[0,1], "k--", linewidth=1, label="Random")
        axes[0].set_xlabel("False Positive Rate", fontsize=11)
        axes[0].set_ylabel("True Positive Rate", fontsize=11)
        axes[0].set_title(f"ROC Curve — {args.model_name}\n"
                          f"Layers {args.critical_layers}",
                          fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Precision-Recall curve
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_probs)
        axes[1].plot(rec_arr, prec_arr, color="#2196F3", linewidth=2)
        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].set_title("Precision-Recall Curve\n"
                          "(Higher = better detector)",
                          fontsize=11, fontweight="bold")
        axes[1].grid(alpha=0.3)

        # Threshold calibration by industry
        industries = list(calib["industry"].keys())
        thresholds = [calib["industry"][i]["threshold"] for i in industries]
        recalls    = [calib["industry"][i]["recall"]    for i in industries]
        precisions = [calib["industry"][i]["precision"] for i in industries]

        x = np.arange(len(industries))
        w = 0.35
        axes[2].bar(x - w/2, recalls,    w, label="Recall",    color="#4CAF50", alpha=0.8)
        axes[2].bar(x + w/2, precisions, w, label="Precision", color="#FF5722", alpha=0.8)
        for i, (ind, t) in enumerate(zip(industries, thresholds)):
            axes[2].text(i, max(recalls[i], precisions[i]) + 0.02,
                        f"τ={t:.2f}", ha="center", fontsize=9)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(industries, fontsize=10)
        axes[2].set_ylabel("Score", fontsize=11)
        axes[2].set_ylim(0, 1.15)
        axes[2].set_title("Industry Threshold Calibration\n"
                          "τ = detection threshold",
                          fontsize=11, fontweight="bold")
        axes[2].legend(fontsize=10)
        axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Lie Detector — {args.model_name} | "
            f"Layers {args.critical_layers} | "
            f"AUC={test_auc:.3f} | F1={test_f1:.3f}",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / "detector_results.png", dpi=150,
                    bbox_inches="tight")
        print(f"Plot saved → {out_dir}/detector_results.png")
        plt.close()

    except Exception as e:
        print(f"Plot failed: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" LIE DETECTOR SUMMARY — {args.model_name.upper()}")
    print(f"{'='*65}")
    print(f" Layers used:      {args.critical_layers}")
    print(f" CV AUC:           {cv_euc['auc']:.3f} ± {cv_euc['auc_std']:.3f}")
    print(f" Test AUC:         {test_auc:.3f}")
    print(f" Test F1:          {test_f1:.3f}")
    print(f" Test Recall:      {test_rec:.3f}")
    print(f" Test Precision:   {test_prec:.3f}")
    print(f"")
    print(f" Healthcare threshold:  {calib['industry']['healthcare']['threshold']:.2f}"
          f" → Recall={calib['industry']['healthcare']['recall']:.3f}")
    print(f" Legal threshold:       {calib['industry']['legal']['threshold']:.2f}"
          f" → F1={calib['industry']['legal']['f1']:.3f}")
    print(f" Finance threshold:     {calib['industry']['finance']['threshold']:.2f}"
          f" → Precision={calib['industry']['finance']['precision']:.3f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()