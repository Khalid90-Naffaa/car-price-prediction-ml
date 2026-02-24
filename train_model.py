"""
Train a price prediction model for sports cars.
Saves: model, predictions, evaluation metrics (separate file), and visualizations (separate files).
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent / "Sport car price.csv"
OUT_DIR = Path(__file__).parent / "model_artifacts"
VIZ_DIR = OUT_DIR / "visualizations"
OUT_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)


def parse_numeric(series):
    """Convert series to numeric, handling commas and + suffixes."""
    s = series.astype(str).str.replace(",", "").str.replace("+", "")
    s = s.replace("", np.nan).replace("N/A", np.nan).replace("nan", np.nan)
    s = s.str.replace(r"^\s*<\s*", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Price (in USD)"] = parse_numeric(df["Price (in USD)"])
    df = df.dropna(subset=["Price (in USD)"])
    df = df[df["Price (in USD)"] > 0]

    engine = df["Engine Size (L)"].astype(str)
    engine = engine.str.extract(r"([\d.]+)", expand=False)
    df["Engine_Size_Num"] = pd.to_numeric(engine, errors="coerce")
    df["Engine_Size_Num"] = df["Engine_Size_Num"].fillna(df["Engine_Size_Num"].median())

    df["Horsepower"] = parse_numeric(df["Horsepower"])
    df["Torque (lb-ft)"] = parse_numeric(df["Torque (lb-ft)"])
    df["0-60 MPH Time (seconds)"] = parse_numeric(df["0-60 MPH Time (seconds)"])
    for col in ["Horsepower", "Torque (lb-ft)", "0-60 MPH Time (seconds)"]:
        df[col] = df[col].fillna(df[col].median())

    le_make = LabelEncoder()
    df["Make_Encoded"] = le_make.fit_transform(df["Car Make"].astype(str))
    return df, le_make


def save_metrics_to_file(metrics, test_df, out_dir=OUT_DIR):
    """Save evaluation metrics to separate JSON and CSV files."""
    # JSON: full metrics (serializable floats)
    metrics_serializable = {
        "train": {k: float(v) for k, v in metrics["train"].items()},
        "test": {k: float(v) for k, v in metrics["test"].items()},
    }
    mape = (np.abs(test_df["Difference"]) / test_df["Actual_Price"]).mean() * 100
    metrics_serializable["test"]["mape_percent"] = float(mape)

    with open(out_dir / "model_evaluation_metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    # CSV: summary table for readability
    summary = pd.DataFrame([
        {"split": "train", "MAE_USD": metrics["train"]["mae"], "RMSE_USD": metrics["train"]["rmse"], "R2": metrics["train"]["r2"]},
        {"split": "test", "MAE_USD": metrics["test"]["mae"], "RMSE_USD": metrics["test"]["rmse"], "R2": metrics["test"]["r2"]},
    ])
    summary.to_csv(out_dir / "model_evaluation_metrics.csv", index=False)
    return metrics_serializable


def save_visualizations(test_df, viz_dir=VIZ_DIR):
    """Generate and save all evaluation visualizations to image files."""
    # 1) Actual vs Predicted — scatter
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(test_df["Actual_Price"], test_df["Predicted_Price"], alpha=0.6, s=40)
    max_p = max(test_df["Actual_Price"].max(), test_df["Predicted_Price"].max())
    ax1.plot([0, max_p], [0, max_p], "k--", lw=2, label="Perfect prediction")
    ax1.set_xlabel("Actual price (USD)")
    ax1.set_ylabel("Predicted price (USD)")
    ax1.set_title("Actual vs Predicted Prices (test set)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(viz_dir / "01_actual_vs_predicted_scatter.png", dpi=120)
    plt.close(fig1)

    # 2) Actual vs Predicted — line chart (first 50)
    sample = test_df.head(50)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    x = range(len(sample))
    ax2.plot(x, sample["Actual_Price"], "o-", label="Actual", color="C0", markersize=4)
    ax2.plot(x, sample["Predicted_Price"], "s-", label="Predicted", color="C1", markersize=4)
    ax2.set_xlabel("Car index")
    ax2.set_ylabel("Price (USD)")
    ax2.set_title("Actual vs Predicted — first 50 test cars")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(viz_dir / "02_actual_vs_predicted_line.png", dpi=120)
    plt.close(fig2)

    # 3) Bar — mean absolute error by brand
    err_by_make = test_df.groupby("Car Make")["Abs_Error"].mean().sort_values(ascending=False).head(15)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    err_by_make.plot(kind="barh", ax=ax3, color="steelblue", alpha=0.8)
    ax3.set_xlabel("Mean absolute error (USD)")
    ax3.set_title("Average prediction error by brand (top 15)")
    fig3.tight_layout()
    fig3.savefig(viz_dir / "03_error_by_brand_bar.png", dpi=120)
    plt.close(fig3)

    # 4) Bar — prediction difference per car (first 40)
    sample40 = test_df.head(40)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    colors = ["green" if d >= 0 else "red" for d in sample40["Difference"]]
    ax4.bar(range(len(sample40)), sample40["Difference"], color=colors, alpha=0.7)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xlabel("Car index")
    ax4.set_ylabel("Difference (Predicted − Actual, USD)")
    ax4.set_title("Over/under prediction per car (first 40)")
    fig4.tight_layout()
    fig4.savefig(viz_dir / "04_prediction_difference_bars.png", dpi=120)
    plt.close(fig4)

    # 5) Residuals distribution
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.hist(test_df["Difference"], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax5.axvline(0, color="black", linewidth=1)
    ax5.set_xlabel("Prediction difference (USD)")
    ax5.set_ylabel("Count")
    ax5.set_title("Distribution of prediction errors")
    fig5.tight_layout()
    fig5.savefig(viz_dir / "05_residuals_histogram.png", dpi=120)
    plt.close(fig5)


def train_and_evaluate():
    df, le_make = load_and_clean()
    feature_cols = [
        "Year", "Engine_Size_Num", "Horsepower", "Torque (lb-ft)",
        "0-60 MPH Time (seconds)", "Make_Encoded",
    ]
    X = df[feature_cols]
    y = df["Price (in USD)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    metrics = {
        "train": {
            "mae": mean_absolute_error(y_train, y_pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "r2": r2_score(y_train, y_pred_train),
        },
        "test": {
            "mae": mean_absolute_error(y_test, y_pred_test),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "r2": r2_score(y_test, y_pred_test),
        },
    }

    # Save model and preprocessing artifacts
    joblib.dump(model, OUT_DIR / "model.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    joblib.dump(le_make, OUT_DIR / "label_encoder_make.joblib")
    joblib.dump(feature_cols, OUT_DIR / "feature_cols.joblib")

    # Predictions in a new file
    test_df = df.loc[y_test.index].copy()
    test_df = test_df[["Car Make", "Car Model", "Year"] + feature_cols + ["Price (in USD)"]]
    test_df = test_df.rename(columns={"Price (in USD)": "Actual_Price"})
    test_df["Predicted_Price"] = y_pred_test
    test_df["Difference"] = test_df["Predicted_Price"] - test_df["Actual_Price"]
    test_df["Abs_Error"] = np.abs(test_df["Difference"])
    test_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Save evaluation metrics to separate files
    save_metrics_to_file(metrics, test_df)
    joblib.dump(metrics, OUT_DIR / "metrics.joblib")

    # Save all visualizations to separate image files
    save_visualizations(test_df)

    return metrics, test_df, feature_cols


if __name__ == "__main__":
    metrics, test_df, _ = train_and_evaluate()
    print("Test metrics:", metrics["test"])
    print("Saved:")
    print("  - Model & artifacts:", OUT_DIR)
    print("  - Predictions:", OUT_DIR / "test_predictions.csv")
    print("  - Evaluation metrics:", OUT_DIR / "model_evaluation_metrics.json", "and", OUT_DIR / "model_evaluation_metrics.csv")
    print("  - Visualizations:", VIZ_DIR)
