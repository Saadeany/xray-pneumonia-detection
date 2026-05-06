from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(y_true, y_pred):

    print("\n" + "=" * 40)
    print("FINAL EVALUATION METRICS")
    print("=" * 40)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"])
    print(report)

    return {
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    }