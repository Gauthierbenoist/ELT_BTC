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
from elt_btc.ml.backtest import (
    max_drawdown,
    meta_effective_proba,
    sharpe_ratio,
    strategy_returns,
)
from elt_btc.ml.runs import Run, list_runs, load_run
from elt_btc.ml.trade_backtest import simulate_trades, simulate_trades_trailing

MS_PER_YEAR = 365 * 86_400 * 1000
RUNS_ROOT = Path("outputs/benchmark")
BUY_HOLD = "buy & hold"
PALETTE = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3", "#ff6692"]

st.set_page_config(page_title="ELT_BTC — benchmark", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Chargement du run…")
def get_run(path_str: str) -> Run:
    return load_run(Path(path_str))


def compute_model(
    df: pd.DataFrame,
    p_up: np.ndarray,
    fee_rate: float,
    band: float,
    bars_per_year: float,
    is_meta: bool = False,
) -> dict[str, object]:
    """All curves and metrics of one strategy on the OOS window.

    For meta-labeling runs ``p_up`` is a win probability: it is mapped to a
    directional probability along the primary side (never fading it) and
    ``ret_next`` (side-adjusted) is mapped back to price space.
    """
    ret_next = df["ret_next"].to_numpy()
    y = df["y"].to_numpy()
    if is_meta:
        side = df["side"].to_numpy()
        p_directional = meta_effective_proba(p_up, side)
        ret_next = side * ret_next
    else:
        p_directional = p_up
    gross, net, positions = strategy_returns(
        p_directional, ret_next, fee_rate=fee_rate, threshold_band=band
    )
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
    is_meta = run.report["config"].get("target", {}).get("type") == "meta_triple_barrier"

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
        computed[name] = compute_model(
            df, df["p_up"].to_numpy(), fee_rate, band, bars_per_year, is_meta=is_meta
        )
        colors[name] = PALETTE[i % len(PALETTE)]
    # Buy & hold reference: constant long on the same OOS grid (one entry fee).
    reference = run.predictions.loc[run.predictions["model"] == selected[0]]
    reference = reference.sort_values("timestamp").reset_index(drop=True)
    if is_meta:
        # In price space: undo the side adjustment before holding long.
        reference = reference.assign(
            ret_next=reference["side"].to_numpy() * reference["ret_next"].to_numpy()
        )
    computed[BUY_HOLD] = compute_model(
        reference, np.ones(len(reference)), fee_rate, band, bars_per_year
    )
    colors[BUY_HOLD] = "#7f7f7f"

    st.caption(
        "⚠️ Tableau = politique **par bougie** (position à chaque bougie selon p) : "
        "utile pour comparer les modèles, mais il sous-compte les frais pour les "
        "labels à barrières. La politique **exécutable** (un trade à la fois, frais "
        "à l'aller-retour) est dans l'onglet *Trades* — c'est elle qui fait foi."
    )
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

    tab_equity, tab_dd, tab_dist, tab_trades, tab_imp, tab_cal = st.tabs(
        ["Équité", "Drawdown", "Distribution", "Trades", "Importances", "Calibration"]
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

    with tab_trades:
        if "holding_bars" not in run.predictions.columns:
            st.info(
                "Ce run ne contient pas la colonne holding_bars "
                "(régénérez-le avec une version récente du benchmark)."
            )
        else:
            bar_ms = timeframe_to_ms(timeframe)
            # The execution policy is a property of the run's configuration.
            configured_trailing = (
                run.report["config"].get("backtest", {}).get("policy") == "trailing"
            )
            use_trailing = (
                configured_trailing
                and is_meta
                and "sigma" in run.predictions.columns
                and run.bars is not None
            )
            if configured_trailing and not use_trailing:
                st.warning(
                    "Politique trailing configurée mais artefacts manquants "
                    "(sigma/bars) : repli sur les barrières fixes."
                )
            target_cfg = run.report["config"].get("target", {})
            trade_rows = {}
            trade_results: dict[str, pd.DataFrame] = {}
            entry_prices: dict[str, pd.Series] = {}
            fig = go.Figure()
            for name in selected:
                df = run.predictions.loc[run.predictions["model"] == name]
                df = df.sort_values("timestamp").reset_index(drop=True)
                if use_trailing:
                    assert run.bars is not None
                    aligned = (
                        run.bars.set_index("timestamp")
                        .reindex(df["timestamp"].to_numpy())
                        .reset_index()
                    )
                    result = simulate_trades_trailing(
                        df["timestamp"].to_numpy(),
                        df["p_up"].to_numpy(),
                        df["side"].to_numpy(),
                        aligned["high"].to_numpy(),
                        aligned["low"].to_numpy(),
                        aligned["close"].to_numpy(),
                        df["sigma"].to_numpy(),
                        bar_ms=bar_ms,
                        fee_rate=fee_rate,
                        pt_mult=float(target_cfg.get("pt_mult", 1.0)),
                        sl_mult=float(target_cfg.get("sl_mult", 1.0)),
                        max_holding=int(target_cfg.get("max_holding", 42)),
                        threshold_band=band,
                    )
                else:
                    result = simulate_trades(
                        df["timestamp"].to_numpy(),
                        df["p_up"].to_numpy(),
                        df["ret_next"].to_numpy(),
                        df["holding_bars"].to_numpy(),
                        bar_ms=bar_ms,
                        fee_rate=fee_rate,
                        threshold_band=band,
                        side=df["side"].to_numpy() if is_meta and "side" in df.columns else None,
                    )
                trade_results[name] = result.trades
                if "close" in df.columns:
                    entry_prices[name] = df.set_index("timestamp")["close"]
                m = result.metrics
                trade_rows[name] = {
                    "Nb trades": m["n_trades"],
                    "Trades/an": m["trades_per_year"],
                    "Win rate": m["win_rate"],
                    "Rdt moyen/trade": m["avg_ret_net"],
                    "Rdt annuel net": m["ann_return_net"],
                    "Sharpe net": m["sharpe_net"],
                    "Max drawdown": m["max_drawdown_net"],
                    "Exposition": m["exposure"],
                    "Holding moyen (bougies)": m["avg_holding_bars"],
                }
                if len(result.trades):
                    fig.add_trace(
                        go.Scattergl(
                            x=pd.to_datetime(result.trades["entry_ts"], unit="ms", utc=True),
                            y=np.cumprod(1.0 + result.trades["ret_net"].to_numpy()),
                            name=name,
                            mode="lines",
                            line={"color": colors[name], "shape": "hv"},
                        )
                    )
            policy_label = (
                "barrières **trailing** (TP/SL ratchetés à chaque re-signal du modèle)"
                if use_trailing
                else "barrières **fixes** (posées à l'entrée)"
            )
            st.caption(
                f"Politique exécutable de ce run : {policy_label} — un seul trade à la fois, "
                "entré si le signal sort de la zone neutre, frais comptés à l'aller-retour. "
                "Les signaux pendant un trade ouvert sont ignorés (ou servent aux mises à "
                "jour de barrières en trailing)."
            )
            st.dataframe(
                pd.DataFrame(trade_rows).T,
                width="stretch",
                column_config={
                    "Nb trades": st.column_config.NumberColumn(format="%.0f"),
                    "Trades/an": st.column_config.NumberColumn(format="%.0f"),
                    "Win rate": st.column_config.NumberColumn(format="%.4f"),
                    "Rdt moyen/trade": st.column_config.NumberColumn(format="percent"),
                    "Rdt annuel net": st.column_config.NumberColumn(format="percent"),
                    "Sharpe net": st.column_config.NumberColumn(format="%.2f"),
                    "Max drawdown": st.column_config.NumberColumn(format="percent"),
                    "Exposition": st.column_config.NumberColumn(format="%.2f"),
                    "Holding moyen (bougies)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            fig.update_layout(
                height=440,
                hovermode="x unified",
                yaxis_title="Équité par trade (base 1)",
                yaxis_type="log" if log_scale else "linear",
                legend={"orientation": "h"},
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("Détail des trades")
            detail_model = st.selectbox(
                "Modèle", [m for m in selected if m in trade_results], key="detail_model"
            )
            trades = trade_results.get(detail_model, pd.DataFrame())
            if trades.empty:
                st.info("Aucun trade avec ces réglages.")
            else:
                detail = trades.copy()
                detail["Entrée"] = pd.to_datetime(detail["entry_ts"], unit="ms", utc=True)
                detail["Sortie"] = pd.to_datetime(detail["exit_ts"], unit="ms", utc=True)
                detail["Sens"] = np.where(detail["direction"] > 0, "Long", "Short")
                detail["Résultat"] = np.where(detail["ret_net"] > 0, "Gagnant", "Perdant")
                if "entry_price" in detail.columns:  # v2: prices recorded directly
                    detail["Prix entrée"] = detail["entry_price"]
                    detail["Prix sortie"] = detail["exit_price"]
                elif detail_model in entry_prices:
                    detail["Prix entrée"] = (
                        detail["entry_ts"].map(entry_prices[detail_model]).astype(float)
                    )
                    detail["Prix sortie"] = detail["Prix entrée"] * (
                        1.0 + detail["direction"] * detail["ret_gross"]
                    )
                else:
                    detail["Prix entrée"] = np.nan  # run antérieur à la colonne close
                    detail["Prix sortie"] = np.nan

                col_result, col_side = st.columns(2)
                with col_result:
                    result_filter = st.radio(
                        "Résultat", ["Tous", "Gagnants", "Perdants"], horizontal=True
                    )
                with col_side:
                    side_filter = st.radio("Sens", ["Tous", "Long", "Short"], horizontal=True)
                if result_filter != "Tous":
                    detail = detail[detail["Résultat"] == result_filter[:-1]]
                if side_filter != "Tous":
                    detail = detail[detail["Sens"] == side_filter]
                st.caption(f"{len(detail)} trade(s) après filtre")

                display_cols = [
                    "Entrée",
                    "Sortie",
                    "Sens",
                    "p_up",
                    "Prix entrée",
                    "Prix sortie",
                    "holding_bars",
                    "ret_gross",
                    "ret_net",
                    "Résultat",
                ]
                if "exit_reason" in detail.columns:  # v2 extras
                    display_cols += ["exit_reason", "n_updates"]
                st.dataframe(
                    detail[display_cols].reset_index(drop=True),
                    width="stretch",
                    column_config={
                        "exit_reason": st.column_config.TextColumn("Sortie via"),
                        "n_updates": st.column_config.NumberColumn("MàJ barrières", format="%.0f"),
                        "Entrée": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                        "Sortie": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                        "p_up": st.column_config.NumberColumn("p ML", format="%.3f"),
                        "Prix entrée": st.column_config.NumberColumn(format="dollar"),
                        "Prix sortie": st.column_config.NumberColumn(format="dollar"),
                        "holding_bars": st.column_config.NumberColumn(
                            "Durée (bougies)", format="%.0f"
                        ),
                        "ret_gross": st.column_config.NumberColumn("Rdt brut", format="percent"),
                        "ret_net": st.column_config.NumberColumn("Rdt net", format="percent"),
                    },
                )

                if len(detail):
                    st.markdown("**Trade par trade**")
                    idx = st.number_input(
                        "Trade n° (dans la liste filtrée)",
                        min_value=1,
                        max_value=len(detail),
                        value=1,
                        step=1,
                    )
                    row = detail.iloc[int(idx) - 1]
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Sens", str(row["Sens"]))
                    c2.metric("p ML", f"{row['p_up']:.3f}")
                    c3.metric(
                        "Entrée",
                        "—" if pd.isna(row["Prix entrée"]) else f"{row['Prix entrée']:,.0f} $",
                        str(row["Entrée"])[:16],
                    )
                    c4.metric(
                        "Sortie",
                        "—" if pd.isna(row["Prix sortie"]) else f"{row['Prix sortie']:,.0f} $",
                        str(row["Sortie"])[:16],
                    )
                    c5.metric(
                        "Rendement net",
                        f"{100 * row['ret_net']:+.2f} %",
                        f"{int(row['holding_bars'])} bougies",
                    )

                    if run.bars is None:
                        st.info(
                            "Vue chandelier indisponible : ce run ne contient pas "
                            "bars.parquet (relancez-le avec une version récente)."
                        )
                    else:
                        candles_before, candles_after = 25, 10
                        entry_ts = int(row["entry_ts"])
                        exit_ts = int(row["exit_ts"])
                        window = run.bars[
                            (run.bars["timestamp"] >= entry_ts - candles_before * bar_ms)
                            & (run.bars["timestamp"] <= exit_ts + candles_after * bar_ms)
                        ]
                        times = pd.to_datetime(window["timestamp"], unit="ms", utc=True)
                        chart = go.Figure(
                            go.Candlestick(
                                x=times,
                                open=window["open"],
                                high=window["high"],
                                low=window["low"],
                                close=window["close"],
                                name="BTC/USDT",
                                showlegend=False,
                            )
                        )
                        is_long = row["direction"] > 0
                        entry_time = pd.to_datetime(entry_ts, unit="ms", utc=True)
                        exit_time = pd.to_datetime(exit_ts, unit="ms", utc=True)
                        chart.add_vrect(
                            x0=entry_time,
                            x1=exit_time,
                            fillcolor="#636efa",
                            opacity=0.10,
                            line_width=0,
                        )
                        if not pd.isna(row["Prix entrée"]):
                            chart.add_trace(
                                go.Scatter(
                                    x=[entry_time],
                                    y=[row["Prix entrée"]],
                                    mode="markers",
                                    name=f"Entrée {row['Sens']}",
                                    marker={
                                        "symbol": "triangle-up" if is_long else "triangle-down",
                                        "size": 14,
                                        "color": "#00cc96" if is_long else "#ef553b",
                                        "line": {"width": 1, "color": "black"},
                                    },
                                )
                            )
                            chart.add_trace(
                                go.Scatter(
                                    x=[exit_time],
                                    y=[row["Prix sortie"]],
                                    mode="markers",
                                    name="Sortie",
                                    marker={
                                        "symbol": "x",
                                        "size": 12,
                                        "color": "#ab63fa",
                                        "line": {"width": 1, "color": "black"},
                                    },
                                )
                            )
                            chart.add_hline(
                                y=row["Prix entrée"], line_dash="dot", line_color="#7f7f7f"
                            )
                            chart.add_hline(
                                y=row["Prix sortie"],
                                line_dash="dot",
                                line_color="#00cc96" if row["ret_net"] > 0 else "#ef553b",
                            )
                        chart.update_layout(
                            height=460,
                            xaxis_rangeslider_visible=False,
                            yaxis_title="Prix ($)",
                            legend={"orientation": "h"},
                            margin={"t": 20},
                        )
                        st.plotly_chart(chart, width="stretch")

                    contributions = run.contributions
                    if contributions is None:
                        st.info(
                            "Contributions par trade indisponibles : régénérez le run "
                            "avec une version récente du benchmark."
                        )
                    else:
                        mask = (contributions["model"] == detail_model) & (
                            contributions["timestamp"] == int(row["entry_ts"])
                        )
                        matched = contributions.loc[mask]
                        if matched.empty:
                            st.info("Pas de contributions pour ce modèle (LightGBM uniquement).")
                        else:
                            contrib_row = (
                                matched.iloc[0].drop(["model", "timestamp", "bias"]).astype(float)
                            )
                            top_contrib = contrib_row.reindex(
                                contrib_row.abs().sort_values(ascending=False).index[:8]
                            )
                            contrib_labels = [str(feat) for feat in top_contrib.index]
                            if run.features is not None:
                                feat_match = run.features.loc[
                                    run.features["timestamp"] == int(row["entry_ts"])
                                ]
                                if len(feat_match):
                                    feature_values = feat_match.iloc[0]
                                    contrib_labels = [
                                        f"{feat} = {feature_values[feat]:.4g}"
                                        if feat in feature_values
                                        else str(feat)
                                        for feat in top_contrib.index
                                    ]
                            bar_fig = go.Figure(
                                go.Bar(
                                    x=top_contrib.to_numpy()[::-1],
                                    y=contrib_labels[::-1],
                                    orientation="h",
                                    marker={
                                        "color": [
                                            "#00cc96" if v > 0 else "#ef553b"
                                            for v in top_contrib.to_numpy()[::-1]
                                        ]
                                    },
                                )
                            )
                            bar_fig.update_layout(
                                height=340,
                                title=f"Pourquoi p = {row['p_up']:.3f} — top 8 contributions",
                                xaxis_title="Contribution (log-odds)",
                                margin={"t": 40},
                            )
                            st.plotly_chart(bar_fig, width="stretch")
                            st.caption(
                                "Contributions TreeSHAP du modèle du fold qui a produit cette "
                                "prédiction. Vert : pousse vers « le trade gagne » ; rouge : "
                                "vers « il perd ». Unités en log-odds (l'espace interne du "
                                "modèle), pas en points de probabilité."
                            )

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
