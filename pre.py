import pandas as pd
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from ts import (
    get_train_test_data,
    run_kaplan_meier,
    run_cox_ph_model,
    run_random_forest_survival,
    run_deepsurv,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np

# Dictionaries to store scores
model_scores = {}
model_metrics = {}

# ---------- Evaluation Helpers ----------
def evaluate_model(predictions, durations, events):
    """Compute concordance index."""
    return concordance_index(durations, predictions, events)

def plot_confusion_and_metrics(predictions, durations, events, model_name):
    """Binary metrics (just for comparison)."""
    pred_binary = (predictions >= np.median(predictions)).astype(int)
    true_binary = events.astype(int)

    cm = confusion_matrix(true_binary, pred_binary)
    print(f"\nConfusion Matrix for {model_name}:\n{cm}")
    print(f"\nClassification Report for {model_name}:\n{classification_report(true_binary, pred_binary, zero_division=0)}")

    # Save metrics for bar chart
    model_metrics[model_name] = {
        "Accuracy": round(accuracy_score(true_binary, pred_binary) * 100, 2),
        "Precision": round(precision_score(true_binary, pred_binary, zero_division=0) * 100, 2),
        "Recall": round(recall_score(true_binary, pred_binary, zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(true_binary, pred_binary, zero_division=0) * 100, 2),
    }

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()

# ---------- Evaluate All Models ----------
def evaluate_all_models():
    # Get train/test split first
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = get_train_test_data()

    models = {
        "Kaplan-Meier": lambda: run_kaplan_meier(y_time_test, y_event_test),
        "Cox Proportional Hazards": lambda: run_cox_ph_model(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test),
        "Random Forest Survival": lambda: run_random_forest_survival(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test),
        "DeepSurv": lambda: run_deepsurv(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test),
    }

    for model_name, model_func in models.items():
        try:
            predictions, durations, events = model_func()
            c_index = evaluate_model(predictions, durations, events)
            model_scores[model_name] = round(c_index * 100, 2)
            print(f"{model_name} C-index: {c_index:.4f} ({c_index * 100:.2f}%)")

            # Metrics + confusion matrix
            plot_confusion_and_metrics(predictions, durations, events, model_name)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

# ---------- Plotting ----------
def plot_model_scores(scores):
    plt.figure(figsize=(10, 6))
    names = list(scores.keys())
    values = list(scores.values())
    bars = plt.bar(names, values, color="skyblue", edgecolor="black")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha="center", va="bottom")

    plt.ylabel("C-index (%)")
    plt.title("Model Prediction Accuracy (C-index)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_prediction_accuracy.png")
    plt.show()

def plot_all_metrics(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.plot(kind="bar", figsize=(12, 7), colormap="Set3", edgecolor="black")

    plt.title("Model Performance Metrics (Test Set)")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("model_metrics_comparison.png")
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    evaluate_all_models()
    plot_model_scores(model_scores)
    plot_all_metrics(model_metrics)
