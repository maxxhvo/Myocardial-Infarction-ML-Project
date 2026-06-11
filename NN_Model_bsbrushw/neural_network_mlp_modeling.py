from pathlib import Path
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

RANDOM_STATE = 321
TARGET = "ZSN"
SPLIT_COLUMN = "train_dummy"
POSITIVE_LABEL = "CHF"
NEGATIVE_LABEL = "No CHF"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Neural_Network_Modeling"
FIGURE_DIR = OUTPUT_DIR / "Figures"
TABLE_DIR = OUTPUT_DIR / "Tables"
MODEL_DIR = OUTPUT_DIR / "Models"


def find_model_data_path() -> Path:
    candidate_paths = [
        PROJECT_ROOT / "Finalized_Feature_Sets" / "model_data_simple.parquet",
        PROJECT_ROOT / "Random_Forest_Modeling" / "model_data_simple.parquet",
        PROJECT_ROOT / "Random_Forest_Modelling" / "model_data_simple.parquet",
        PROJECT_ROOT / "model_data_simple.parquet",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find model_data_simple.parquet.")


def load_data():
    data_path = find_model_data_path()
    model_data = pd.read_parquet(data_path)
    feature_cols = model_data.columns.drop([TARGET, SPLIT_COLUMN])

    x_train = model_data.loc[model_data[SPLIT_COLUMN] == 1, feature_cols].copy()
    y_train = model_data.loc[model_data[SPLIT_COLUMN] == 1, TARGET].astype(int).copy()
    x_test = model_data.loc[model_data[SPLIT_COLUMN] == 0, feature_cols].copy()
    y_test = model_data.loc[model_data[SPLIT_COLUMN] == 0, TARGET].astype(int).copy()

    categorical_cols = x_train.select_dtypes(
        include=["category", "object", "bool"]
    ).columns.tolist()

    return data_path, model_data, x_train, y_train, x_test, y_test, categorical_cols


def dummy_encode_features(x_train, x_test, categorical_cols):
    x_train_encoded = pd.get_dummies(
        x_train,
        columns=categorical_cols,
        drop_first=True,
        dtype=float,
    )
    x_test_encoded = pd.get_dummies(
        x_test,
        columns=categorical_cols,
        drop_first=True,
        dtype=float,
    )
    x_train_encoded, x_test_encoded = x_train_encoded.align(
        x_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )
    return x_train_encoded.astype(float), x_test_encoded.astype(float)


def build_initial_mlp():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_search():
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    batch_size=32,
                    early_stopping=True,
                    validation_fraction=0.20,
                    n_iter_no_change=20,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_distributions = {
        "mlp__hidden_layer_sizes": [
            (16,),
            (32,),
            (64,),
            (32, 16),
            (64, 32),
            (64, 32, 16),
        ],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": loguniform(1e-5, 1e-1),
        "mlp__learning_rate_init": loguniform(1e-4, 1e-2),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
    )


def positive_probabilities(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x).astype(float)


def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_prob = positive_probabilities(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "log_loss": log_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def format_classification_report_with_micro(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
        output_dict=True,
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="micro",
        zero_division=0,
    )

    total_support = int(report[NEGATIVE_LABEL]["support"] + report[POSITIVE_LABEL]["support"])
    rows = [
        (
            NEGATIVE_LABEL,
            report[NEGATIVE_LABEL]["precision"],
            report[NEGATIVE_LABEL]["recall"],
            report[NEGATIVE_LABEL]["f1-score"],
            int(report[NEGATIVE_LABEL]["support"]),
        ),
        (
            POSITIVE_LABEL,
            report[POSITIVE_LABEL]["precision"],
            report[POSITIVE_LABEL]["recall"],
            report[POSITIVE_LABEL]["f1-score"],
            int(report[POSITIVE_LABEL]["support"]),
        ),
        ("accuracy", None, None, report["accuracy"], total_support),
        ("micro avg", micro_precision, micro_recall, micro_f1, total_support),
        (
            "macro avg",
            report["macro avg"]["precision"],
            report["macro avg"]["recall"],
            report["macro avg"]["f1-score"],
            int(report["macro avg"]["support"]),
        ),
        (
            "weighted avg",
            report["weighted avg"]["precision"],
            report["weighted avg"]["recall"],
            report["weighted avg"]["f1-score"],
            int(report["weighted avg"]["support"]),
        ),
    ]

    lines = [
        f"{'':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "",
    ]
    for label, precision, recall, f1_value, support in rows:
        if precision is None:
            lines.append(f"{label:>14} {'':>10} {'':>10} {f1_value:>10.2f} {support:>10}")
        else:
            lines.append(
                f"{label:>14} {precision:>10.2f} {recall:>10.2f} {f1_value:>10.2f} {support:>10}"
            )
    return "\n".join(lines)


def format_screenshot_style_results(title, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_prob = positive_probabilities(model, x_test)
    return "\n".join(
        [
            f"=== {title} RESULTS ===",
            f"NN Test Accuracy: {accuracy_score(y_test, y_pred):.4f}",
            "",
            format_classification_report_with_micro(y_test, y_pred),
            "",
            f"NN Test Log Loss: {log_loss(y_test, y_prob):.4f}",
            f"NN Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}",
        ]
    )


def save_classification_reports(models, x_test, y_test):
    report_path = TABLE_DIR / "classification_reports.txt"
    lines = []
    for model_name, model in models.items():
        report = format_screenshot_style_results(model_name, model, x_test, y_test)
        lines.append(report + "\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_plots(models, x_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, model in models.items():
        y_prob = positive_probabilities(model, x_test)
        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            name=model_name,
            ax=ax,
        )
    ax.set_title("ROC Curve for Majority Baseline and MLP Models")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_curve_majority_initial_optimized_mlp.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, model in models.items():
        y_prob = positive_probabilities(model, x_test)
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_prob,
            name=model_name,
            ax=ax,
        )
    ax.set_title("Precision-Recall Curve for Majority Baseline and MLP Models")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_curve_majority_initial_optimized_mlp.png", dpi=300)
    plt.close(fig)

    optimized = models["Optimized MLP"]
    cm = confusion_matrix(y_test, optimized.predict(x_test), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ylabel="True label",
        xlabel="Predicted label",
        title="Optimized MLP Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optimized_mlp_confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_error_analysis(model_data, x_test_raw, x_test_model, y_test, optimized_model):
    y_prob = positive_probabilities(optimized_model, x_test_model)
    y_pred = (y_prob >= 0.5).astype(int)

    key_columns = [
        "AGE",
        "SEX",
        "INF_ANAM",
        "IBS_POST",
        "GB",
        "S_AD_ORIT",
        "D_AD_ORIT",
        "NA_BLOOD",
        "K_BLOOD",
        "L_BLOOD",
        "ROE_log",
        "ALT_BLOOD_log",
        "AST_BLOOD_log",
        "ZSN_A_recode",
        "time_bin",
        "angina_timing",
        "angina_severity",
        "htn_duration",
    ]
    key_columns = [col for col in key_columns if col in x_test_raw.columns]

    error_frame = x_test_raw[key_columns].copy()
    error_frame.insert(0, "patient_index", error_frame.index)
    error_frame.insert(1, "true_ZSN", y_test.values)
    error_frame.insert(2, "predicted_ZSN", y_pred)
    error_frame.insert(3, "predicted_probability_CHF", y_prob)

    conditions = [
        (error_frame["true_ZSN"] == 1) & (error_frame["predicted_ZSN"] == 0),
        (error_frame["true_ZSN"] == 0) & (error_frame["predicted_ZSN"] == 1),
    ]
    error_frame["error_type"] = np.select(
        conditions,
        ["false_negative", "false_positive"],
        default="correct",
    )

    errors = error_frame.loc[error_frame["error_type"] != "correct"].copy()
    errors = errors.sort_values(
        ["error_type", "predicted_probability_CHF"],
        ascending=[True, False],
    )
    errors.to_csv(TABLE_DIR / "optimized_mlp_test_errors.csv", index=False)

    error_summary = (
        error_frame["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    error_summary.to_csv(TABLE_DIR / "optimized_mlp_error_summary.csv", index=False)

    scored_test = model_data.loc[model_data[SPLIT_COLUMN] == 0].copy()
    scored_test["predicted_probability_CHF"] = y_prob
    scored_test["predicted_ZSN"] = y_pred
    scored_test["error_type"] = error_frame["error_type"].values
    scored_test.to_csv(TABLE_DIR / "optimized_mlp_scored_test_set.csv", index=True)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (
        _data_path,
        model_data,
        x_train,
        y_train,
        x_test,
        y_test,
        categorical_cols,
    ) = load_data()
    x_train_model, x_test_model = dummy_encode_features(x_train, x_test, categorical_cols)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(x_train_model, y_train)

    initial_mlp = build_initial_mlp()
    initial_mlp.fit(x_train_model, y_train)

    search = build_search()
    search.fit(x_train_model, y_train, mlp__sample_weight=sample_weight)
    optimized_mlp = search.best_estimator_

    models = {
        "Majority baseline": majority_baseline,
        "Initial MLP": initial_mlp,
        "Optimized MLP": optimized_mlp,
    }

    metrics_df = pd.DataFrame(
        [
            evaluate_model(model_name, model, x_test_model, y_test)
            for model_name, model in models.items()
        ]
    )
    metrics_df.to_csv(TABLE_DIR / "model_performance_metrics.csv", index=False)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["mean_test_log_loss"] = -cv_results["mean_test_score"]
    cv_results["mean_train_log_loss"] = -cv_results["mean_train_score"]
    cv_results = cv_results.sort_values("mean_test_log_loss")
    cv_results.to_csv(TABLE_DIR / "hyperparameter_search_results.csv", index=False)

    save_classification_reports(models, x_test_model, y_test)
    save_plots(models, x_test_model, y_test)
    save_error_analysis(model_data, x_test, x_test_model, y_test, optimized_mlp)
    with (MODEL_DIR / "optimized_mlp_pipeline.pkl").open("wb") as model_file:
        pickle.dump(optimized_mlp, model_file)
    pd.Series(x_train_model.columns, name="feature").to_csv(
        MODEL_DIR / "dummy_encoded_feature_columns.csv",
        index=False,
    )
    print("Saved neural network modeling outputs to:")
    print(f"  {OUTPUT_DIR}")
    print()
    print(format_screenshot_style_results("MAJORITY CLASS BASELINE", majority_baseline, x_test_model, y_test))
    print()
    print(format_screenshot_style_results("SIMPLE NEURAL NETWORK (MLP)", initial_mlp, x_test_model, y_test))
    print()
    print(format_screenshot_style_results("OPTIMIZED NEURAL NETWORK (MLP)", optimized_mlp, x_test_model, y_test))

    print("\nCompact test set performance table:")
    print(metrics_df.to_string(index=False))
    print("\nBest optimized MLP parameters:")
    print(search.best_params_)
    print(f"Best cross-validated log loss: {-search.best_score_:.4f}")


if __name__ == "__main__":
    main()
