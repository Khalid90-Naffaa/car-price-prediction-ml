"""
Sports Car Price Prediction — كل المشروع في ملف واحد (po.py)
- تدريب النموذج:  python po.py
- واجهة Streamlit: streamlit run po.py
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# المسارات (نسباً لموقع po.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Sport car price.csv"
OUT_DIR = BASE_DIR / "model_artifacts"
VIZ_DIR = OUT_DIR / "visualizations"


def parse_numeric(series):
    s = series.astype(str).str.replace(",", "").str.replace("+", "")
    s = s.replace("", np.nan).replace("N/A", np.nan).replace("nan", np.nan)
    s = s.str.replace(r"^\s*<\s*", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def load_and_clean(path=None):
    path = path or DATA_PATH
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


def save_metrics_to_file(metrics, test_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = {
        "train": {k: float(v) for k, v in metrics["train"].items()},
        "test": {k: float(v) for k, v in metrics["test"].items()},
    }
    m["test"]["mape_percent"] = float((np.abs(test_df["Difference"]) / test_df["Actual_Price"]).mean() * 100)
    with open(OUT_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    summary = pd.DataFrame([
        {"split": "train", "MAE_USD": metrics["train"]["mae"], "RMSE_USD": metrics["train"]["rmse"], "R2": metrics["train"]["r2"]},
        {"split": "test", "MAE_USD": metrics["test"]["mae"], "RMSE_USD": metrics["test"]["rmse"], "R2": metrics["test"]["r2"]},
    ])
    summary.to_csv(OUT_DIR / "model_evaluation_metrics.csv", index=False)


def save_visualizations(test_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

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
    fig1.savefig(VIZ_DIR / "01_actual_vs_predicted_scatter.png", dpi=120)
    plt.close(fig1)

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
    fig2.savefig(VIZ_DIR / "02_actual_vs_predicted_line.png", dpi=120)
    plt.close(fig2)

    err_by_make = test_df.groupby("Car Make")["Abs_Error"].mean().sort_values(ascending=False).head(15)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    err_by_make.plot(kind="barh", ax=ax3, color="steelblue", alpha=0.8)
    ax3.set_xlabel("Mean absolute error (USD)")
    ax3.set_title("Average prediction error by brand (top 15)")
    fig3.tight_layout()
    fig3.savefig(VIZ_DIR / "03_error_by_brand_bar.png", dpi=120)
    plt.close(fig3)

    sample40 = test_df.head(40)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    colors = ["green" if d >= 0 else "red" for d in sample40["Difference"]]
    ax4.bar(range(len(sample40)), sample40["Difference"], color=colors, alpha=0.7)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xlabel("Car index")
    ax4.set_ylabel("Difference (Predicted − Actual, USD)")
    ax4.set_title("Over/under prediction per car (first 40)")
    fig4.tight_layout()
    fig4.savefig(VIZ_DIR / "04_prediction_difference_bars.png", dpi=120)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.hist(test_df["Difference"], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax5.axvline(0, color="black", linewidth=1)
    ax5.set_xlabel("Prediction difference (USD)")
    ax5.set_ylabel("Count")
    ax5.set_title("Distribution of prediction errors")
    fig5.tight_layout()
    fig5.savefig(VIZ_DIR / "05_residuals_histogram.png", dpi=120)
    plt.close(fig5)


def train_and_evaluate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

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

    joblib.dump(model, OUT_DIR / "model.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    joblib.dump(le_make, OUT_DIR / "label_encoder_make.joblib")
    joblib.dump(feature_cols, OUT_DIR / "feature_cols.joblib")

    test_df = df.loc[y_test.index].copy()
    test_df = test_df[["Car Make", "Car Model", "Year"] + feature_cols + ["Price (in USD)"]]
    test_df = test_df.rename(columns={"Price (in USD)": "Actual_Price"})
    test_df["Predicted_Price"] = y_pred_test
    test_df["Difference"] = test_df["Predicted_Price"] - test_df["Actual_Price"]
    test_df["Abs_Error"] = np.abs(test_df["Difference"])
    test_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    save_metrics_to_file(metrics, test_df)
    joblib.dump(metrics, OUT_DIR / "metrics.joblib")
    save_visualizations(test_df)

    return metrics, test_df


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Sports Car Price Model", layout="wide")
    st.title("Sports Car Price Prediction — Model Performance")

    if not (OUT_DIR / "model.joblib").exists():
        st.error("النموذج غير موجود. شغّل: python po.py")
        return

    if not (OUT_DIR / "model_evaluation_metrics.json").exists():
        st.error("ملف المقاييس غير موجود. شغّل: python po.py")
        return

    with open(OUT_DIR / "model_evaluation_metrics.json") as f:
        metrics = json.load(f)
    test_df = pd.read_csv(OUT_DIR / "test_predictions.csv")

    st.header("دقة النموذج ومقاييس الأداء")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² (اختبار)", f"{metrics['test']['r2']:.4f}")
    with col2:
        st.metric("MAE (دولار)", f"${metrics['test']['mae']:,.0f}")
    with col3:
        st.metric("RMSE (دولار)", f"${metrics['test']['rmse']:,.0f}")
    with col4:
        st.metric("متوسط الخطأ %", f"{metrics['test'].get('mape_percent', 0):.2f}%")

    st.subheader("مقارنة تدريب / اختبار")
    comparison = pd.DataFrame({
        "Split": ["Train", "Test"],
        "MAE (USD)": [metrics["train"]["mae"], metrics["test"]["mae"]],
        "RMSE (USD)": [metrics["train"]["rmse"], metrics["test"]["rmse"]],
        "R²": [metrics["train"]["r2"], metrics["test"]["r2"]],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.header("الرسوم المحفوظة")
    viz_list = [
        ("01_actual_vs_predicted_scatter.png", "الفعلي vs المتوقع (مبعثر)"),
        ("02_actual_vs_predicted_line.png", "الفعلي vs المتوقع (خط — أول 50)"),
        ("03_error_by_brand_bar.png", "متوسط الخطأ حسب العلامة"),
        ("04_prediction_difference_bars.png", "فرق التوقع لكل سيارة"),
        ("05_residuals_histogram.png", "توزيع الأخطاء"),
    ]
    for filename, title in viz_list:
        path = VIZ_DIR / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"الصورة غير موجودة: {filename}. شغّل: python po.py")

    st.header("جدول التوقعات (فعلي / متوقع / فرق)")
    display_cols = ["Car Make", "Car Model", "Year", "Actual_Price", "Predicted_Price", "Difference"]
    df_display = test_df[display_cols].rename(columns={
        "Actual_Price": "Actual (USD)",
        "Predicted_Price": "Predicted (USD)",
        "Difference": "Difference (USD)",
    })
    st.dataframe(
        df_display.style.format({
            "Actual (USD)": "${:,.0f}",
            "Predicted (USD)": "${:,.0f}",
            "Difference (USD)": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("الفرق = المتوقع − الفعلي. موجب = فوق التقدير، سالب = تحت التقدير.")


def _is_running_in_streamlit():
    t = threading.current_thread()
    return type(t).__module__.startswith("streamlit.")


if __name__ == "__main__":
    if _is_running_in_streamlit():
        run_streamlit_app()
    else:
        metrics, test_df = train_and_evaluate()
        print("Test metrics:", metrics["test"])
        print("تم الحفظ في:")
        print("  - النموذج والتوقعات:", OUT_DIR)
        print("  - المقاييس:", OUT_DIR / "model_evaluation_metrics.json", "و", OUT_DIR / "model_evaluation_metrics.csv")
        print("  - الرسوم:", VIZ_DIR)
