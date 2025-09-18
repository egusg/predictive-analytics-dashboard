
import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import plotly.graph_objects as go
from pathlib import Path
from utils import load_series, train_test_split_ts

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, np.abs(y_true)))) * 100

def naive_seasonal_forecast(train, horizon, season):
    if len(train) < season:
        return np.repeat(train.iloc[-1], horizon)
    last_season = train.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season.values, reps)[:horizon]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample_timeseries.csv")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="value")
    parser.add_argument("--freq", default="D")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--season", type=int, default=7)
    parser.add_argument("--run-name", default="sample", help="Name of the run (results saved in results/<run-name>/)")
    args = parser.parse_args()

    # Load and preprocess
    df = load_series(args.csv, args.date_col, args.target_col)
    df = df.set_index(args.date_col).asfreq(args.freq).interpolate()

    horizon = min(args.horizon, max(7, int(len(df)*0.1)))
    train_df, test_df = df.iloc[:-horizon], df.iloc[-horizon:]
    y_train, y_test = train_df[args.target_col], test_df[args.target_col]

    # Baseline model
    baseline_pred = naive_seasonal_forecast(y_train, horizon, args.season)

    # AutoARIMA
    arima_model = auto_arima(y_train, seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action="ignore")
    arima_pred = arima_model.predict(n_periods=horizon)

    # Metrics
    metrics = {
        "baseline": {
            "MAE": float(mean_absolute_error(y_test, baseline_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
            "MAPE": float(mape(y_test, baseline_pred))
        },
        "auto_arima": {
            "MAE": float(mean_absolute_error(y_test, arima_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, arima_pred))),
            "MAPE": float(mape(y_test, arima_pred))
        }
    }

    # Create output directory for this run
    out_dir = Path("results") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save static forecast plot
    plt.plot(train_df.index, y_train, label="train")
    plt.plot(test_df.index, y_test, label="test")
    plt.plot(test_df.index, baseline_pred, label="naive seasonal")
    plt.plot(test_df.index, arima_pred, label="auto_arima")
    plt.legend(); plt.title("Forecast Comparison")
    plt.tight_layout(); plt.savefig(out_dir / "forecast_plot.png", dpi=160); plt.close()

    # Save interactive dashboard
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=y_train, name="train"))
    fig.add_trace(go.Scatter(x=test_df.index, y=y_test, name="test"))
    fig.add_trace(go.Scatter(x=test_df.index, y=baseline_pred, name="naive seasonal"))
    fig.add_trace(go.Scatter(x=test_df.index, y=arima_pred, name="auto_arima"))
    fig.write_html(str(out_dir / "interactive_forecast.html"), include_plotlyjs="cdn")

    # Save ARIMA summary
    with open(out_dir / "model_summary.txt", "w") as f:
        f.write(str(arima_model.summary()))

    print(f"Done. See {out_dir}/ for outputs.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

