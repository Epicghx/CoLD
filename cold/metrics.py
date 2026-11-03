import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def prevalence_bands(Y_train, head=0.05, mid=0.01):
    """Statement indices by training prevalence: head >= 5%, mid 1-5%, tail < 1%."""
    p = Y_train.mean(0)
    return {"head": np.flatnonzero(p >= head), "mid": np.flatnonzero((p >= mid) & (p < head)),
            "tail": np.flatnonzero(p < mid)}


def multilabel_metrics(probs, Y, Y_train):
    valid = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
    auroc = np.full(Y.shape[1], np.nan)
    auprc = np.full(Y.shape[1], np.nan)
    for k in np.flatnonzero(valid):
        auroc[k] = roc_auc_score(Y[:, k], probs[:, k])
        auprc[k] = average_precision_score(Y[:, k], probs[:, k])
    out = {"macro_auroc": np.nanmean(auroc), "macro_auprc": np.nanmean(auprc)}
    for band, idx in prevalence_bands(Y_train).items():
        out[f"{band}_auprc"] = np.nanmean(auprc[idx]) if len(idx) else float("nan")
    return {k: float(v) for k, v in out.items()}
