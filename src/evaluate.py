from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Prints and returns a full evaluation report.

    Args:
        y_true  : Ground-truth labels (list/array of 0s and 1s)
        y_pred  : Predicted labels after thresholding
        y_prob  : Raw pneumonia probabilities from softmax (needed for ROC-AUC)
    """

    print("\n" + "=" * 50)
    print("         FINAL EVALUATION METRICS")
    print("=" * 50)

    # ── 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print(f"         NORMAL  PNEUMONIA")
    print(f"  NORMAL    {cm[0][0]:>6}  {cm[0][1]:>9}")
    print(f"  PNEUMONIA {cm[1][0]:>6}  {cm[1][1]:>9}")

    # ── 2. Per-Class Report (Precision / Recall / F1)
    print("\nClassification Report:")
    report_str = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"])
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True
    )

    result = {
        "confusion_matrix":  cm.tolist(),
        "report": report_dict,
    }

    # ── 3. ROC-AUC  (requires probabilities, not just predictions)
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC Score : {roc_auc:.4f}")
        result["roc_auc"] = round(roc_auc, 4)
    else:
        print("ROC-AUC       : N/A (pass y_prob to enable)")

    # ── 4. Threshold Tuning  —  find optimal threshold for Recall ≥ 90%
    if y_prob is not None:
        optimal = tune_threshold(y_true, y_prob, target_recall=0.90)
        result["optimal_threshold"] = optimal

    print("=" * 50)
    return result


def tune_threshold(y_true, y_prob, target_recall=0.90):
    """
    Scans the precision-recall curve to find the lowest threshold
    that still achieves the target recall for the PNEUMONIA class.

    Returns the optimal threshold as a float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    print(f"\nThreshold Tuning (target Pneumonia Recall >= {target_recall:.0%}):")
    print(f"  {'Threshold':>10}  {'Recall':>8}  {'Precision':>10}")
    print(f"  {'-'*32}")

    optimal_threshold = 0.5
    found = False

    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if recall >= target_recall:
            print(f"  {threshold:>10.3f}  {recall:>8.3f}  {precision:>10.3f}  <- candidate")
            optimal_threshold = threshold
            found = True

    if not found:
        print(f"  [WARNING] Could not reach Recall >= {target_recall:.0%} at any threshold.")
        print(f"  Defaulting to threshold = 0.40")
        optimal_threshold = 0.40

    print(f"\n  -> Optimal Threshold Selected: {optimal_threshold:.3f}")
    return round(float(optimal_threshold), 4)