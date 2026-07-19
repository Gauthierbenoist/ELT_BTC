"""Streamlit dashboard comparing benchmark models on their OOS predictions.

Everything is recomputed live from ``predictions.parquet`` via the same
``strategy_returns`` used by the benchmark — the fee and neutral-band
sliders therefore re-price every curve and metric instantly, without ever
reloading a model.

Run::

    uv sync --group dashboard --native-tls
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from elt_btc.candles import timeframe_to_ms
from elt_btc.ml.backtest import max_drawdown, sharpe_ratio, strategy_returns
from elt_btc.ml.runs import Run, list_runs, load_run

MS_PER_YEAR = 365 * 86_400 * 1000
RUNS_ROOT = Path("outputs/benchmark")
BUY_HOLD = "buy & hold"
PALETTE = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3", "#ff6692"]

st.set_page_config(page_title="ELT_BTC — benchmark", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Chargement du run…")
def get_run(path_str: str) -> Run:
    return load_run(Path(path_str))


def compute_model(
    df: pd.DataFrame, p_up: np.ndarray, fee_rate: float, band: float, bars_per_year: float
) -> dict[str, object]:
    """All curves and metrics of one strategy on the OOS window."""
    ret_next = df["ret_next"].to_numpy()
    y = df["y"].to_numpy()
    gross, net, positions = strategy_returns(p_up, ret_next, fee_rate=fee_rate, threshold_band=band)
    changes = np.abs(np.diff(positions, prepend=0.0))
    equity = np.cumprod(1.0 + net)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    in_market = positions != 0
    distinct = bool(0.0 < y.mean() < 1.0) and len(np.unique(p_up)) > 1
    return {
        "times": pd.to_datetime(df["timestamp"], unit="ms", utc=True),
        "net": net,
        "equity": equity,
        "drawdown": drawdown,
        "metrics": {
            "Sharpe net": sharpe_ratio(net, bars_per_year),
            "Sharpe brut": sharpe_ratio(gross, bars_per_year),
            "Rdt annuel net": float(net.mean() * bars_per_year),
            "Vol annuelle": float(net.std(ddof=1) * np.sqrt(bars_per_year)),
            "Max drawdown": max_drawdown(net),
            "AUC": float(roc_auc_score(y, p_up)) if distinct else float("nan"),
            "Accuracy": float(((p_up >= 0.5) == (y == 1)).mean()),
            "Hit rate": float((gross[in_market] > 0).mean()) if in_market.any() else float("nan"),
            "Nb trades": float(changes.sum()),
            "Turnover/bougie": float(changes.mean()),
            "Exposition": float(np.abs(positions).mean()),
        },
    }


def main() -> None:
    runs = list_runs(RUNS_ROOT)
    if not runs:
        st.error(
            f"Aucun run sous `{RUNS_ROOT}`. Lancez d'abord `uv run python -m elt_btc.ml.benchmark`."
        )
        st.stop()

    with st.sidebar:
        st.title("ELT_BTC benchmark")
        run_name = st.selectbox("Run", [r.name for r in runs])
        run = get_run(str(RUNS_ROOT / run_name))
        all_models = sorted(run.predictions["model"].unique())
        selected = st.multiselect("Modèles", all_models, default=all_models)
        fee_bps = st.slider("Frais (bps, aller simple)", 0.0, 30.0, 10.0, 0.5)
        band = st.slider("Zone neutre autour de p=0,5", 0.0, 0.20, 0.0, 0.01)
        log_scale = st.toggle("Équité en échelle log")
        st.caption(f"Généré : {run.report.get('generated_at', '?')}")
        commit = run.report.get("git_commit")
        if commit:
            st.caption(f"Commit : `{str(commit)[:10]}`")

    if not selected:
        st.info("Sélectionnez au moins un modèle.")
        st.stop()

    timeframe = str(run.report["config"]["dataset"]["timeframe"])
    bars_per_year = MS_PER_YEAR / timeframe_to_ms(timeframe)
    fee_rate = fee_bps / 10_000.0
    info = run.report.get("dataset", {})

    st.title("Benchmark P(hausse) BTC — analyse out-of-sample")
    st.caption(
        f"{info.get('n_samples', '?')} échantillons {timeframe} · "
        f"{str(info.get('start', '?'))[:10]} → {str(info.get('end', '?'))[:10]} · "
        f"frais {fee_bps:g} bps · zone neutre ±{band:g}"
    )

    computed: dict[str, dict[str, object]] = {}
    colors: dict[str, str] = {}
    for i, name in enumerate(selected):
        df = run.predictions.loc[run.predictions["model"] == name].sort_values("timestamp")
        df = df.reset_index(drop=True)
        computed[name] = compute_model(df, df["p_up"].to_numpy(), fee_rate, band, bars_per_year)
        colors[name] = PALETTE[i % len(PALETTE)]
    # Buy & hold reference: constant long on the same OOS grid (one entry fee).
    reference = run.predictions.loc[run.predictions["model"] == selected[0]]
    reference = reference.sort_values("timestamp").reset_index(drop=True)
    computed[BUY_HOLD] = compute_model(
        reference, np.ones(len(reference)), fee_rate, band, bars_per_year
    )
    colors[BUY_HOLD] = "#7f7f7f"

    table = pd.DataFrame({name: data["metrics"] for name, data in computed.items()}).T
    st.dataframe(
        table,
        width="stretch",
        column_config={
            "Sharpe net": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe brut": st.column_config.NumberColumn(format="%.2f"),
            "Rdt annuel net": st.column_config.NumberColumn(format="percent"),
            "Vol annuelle": st.column_config.NumberColumn(format="percent"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "AUC": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Hit rate": st.column_config.NumberColumn(format="%.4f"),
            "Nb trades": st.column_config.NumberColumn(format="%.0f"),
            "Turnover/bougie": st.column_config.NumberColumn(format="%.3f"),
            "Exposition": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    tab_equity, tab_dd, tab_dist, tab_imp, tab_cal = st.tabs(
        ["Équité", "Drawdown", "Distribution", "Importances", "Calibration"]
    )

    with tab_equity:
        fig = go.Figure()
        for name, data in computed.items():
            fig.add_trace(
                go.Scattergl(
                    x=data["times"],
                    y=data["equity"],
                    name=name,
                    line={"color": colors[name], "dash": "dash" if name == BUY_HOLD else "solid"},
                )
            )
        fig.update_layout(
            height=480,
            hovermode="x unified",
            yaxis_title="Équité (base 1)",
            yaxis_type="log" if log_scale else "linear",
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, width="stretch")

    with tab_dd:
        fig = go.Figure()
        for name, data in computed.items():
            fig.add_trace(
                go.Scattergl(
                    x=data["times"],
                    y=100 * np.asarray(data["drawdown"]),
                    name=name,
                    fill="tozeroy" if name != BUY_HOLD else None,
                    opacity=0.75,
                    line={"color": colors[name], "dash": "dash" if name == BUY_HOLD else "solid"},
                )
            )
        fig.update_layout(
            height=480,
            hovermode="x unified",
            yaxis_title="Drawdown (%)",
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, width="stretch")

    with tab_dist:
        fig = go.Figure()
        stats_rows = {}
        for name, data in computed.items():
            net = np.asarray(data["net"])
            fig.add_trace(
                go.Histogram(
                    x=100 * net,
                    name=name,
                    opacity=0.5,
                    nbinsx=120,
                    marker={"color": colors[name]},
                )
            )
            stats_rows[name] = {
                "Skewness": float(scipy_stats.skew(net)),
                "Kurtosis (excès)": float(scipy_stats.kurtosis(net)),
                "VaR 5% (par bougie)": float(np.quantile(net, 0.05)),
                "Pire bougie": float(net.min()),
                "Meilleure bougie": float(net.max()),
            }
        fig.update_layout(
            barmode="overlay",
            height=440,
            xaxis_title=f"Rendement net par bougie {timeframe} (%)",
            yaxis_title="Fréquence",
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(stats_rows).T.style.format("{:.4f}"), width="stretch")

    with tab_imp:
        with_importances = [m for m in selected if run.importances.get(m)]
        if not with_importances:
            st.info("Pas d'importances pour les modèles sélectionnés (baselines naïves).")
        for name in with_importances:
            top = list(run.importances[name].items())[:15]
            labels = [t[0] for t in reversed(top)]
            values = [t[1] for t in reversed(top)]
            fig = go.Figure(
                go.Bar(x=values, y=labels, orientation="h", marker={"color": colors.get(name)})
            )
            unit = "gain total" if name == "lightgbm" else "|coefficient| (features standardisées)"
            fig.update_layout(height=420, title=f"{name} — top 15 ({unit})")
            st.plotly_chart(fig, width="stretch")

    with tab_cal:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="calibration parfaite",
                line={"dash": "dash", "color": "#7f7f7f"},
            )
        )
        for name in selected:
            bins = run.calibration.get(name, [])
            if not bins:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[b["mean_p_up"] for b in bins],
                    y=[b["realized_up_rate"] for b in bins],
                    name=name,
                    mode="lines+markers",
                    marker={"color": colors[name]},
                )
            )
        fig.update_layout(
            height=480,
            xaxis_title="p(hausse) prédite (moyenne par bin)",
            yaxis_title="Taux de hausse réalisé",
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Calibration calculée sur l'ensemble des prédictions out-of-sample du run "
            "(indépendante des sliders)."
        )


main()
