"""
charts.py — Grafici Plotly standardizzati (tema dark) per la dashboard ciclica.

Ogni funzione restituisce un go.Figure pronto per st.plotly_chart(). Palette e layout
condivisi. I riquadri interpretativi ("come si legge") stanno in app.py, non qui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
    "background": "#1E1E2E",
    "surface": "#2A2A3E",
    "text": "#E0E0E0",
    "accent": "#AB47BC",
}


def _base_layout(title: str, x_title: str = "", y_title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444466"),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=60),
    )


# ============================================================
# 1. PREZZO + OSCILLATORE CICLICO (con zone IS/OOS)
# ============================================================
def build_price_cycle_chart(price: pd.Series, osc: pd.Series, split_date,
                            holdout_date, ticker: str) -> go.Figure:
    """Prezzo (con zone IS/OOS/holdout ombreggiate) sopra, oscillatore ciclico sotto."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Prezzo", "Oscillatore ciclico (normalizzato)"))

    fig.add_trace(go.Scatter(
        x=price.index, y=price.values, name="Prezzo",
        line=dict(color=COLORS["primary"], width=1.4),
        hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)

    # Zone temporali
    fig.add_vrect(x0=price.index[0], x1=split_date, fillcolor=COLORS["primary"],
                  opacity=0.06, line_width=0, row=1, col=1,
                  annotation_text="In-Sample (calibrazione)", annotation_position="top left",
                  annotation_font_color=COLORS["text"])
    fig.add_vrect(x0=split_date, x1=holdout_date, fillcolor=COLORS["secondary"],
                  opacity=0.07, line_width=0, row=1, col=1,
                  annotation_text="Out-of-Sample (test)", annotation_position="top left",
                  annotation_font_color=COLORS["text"])
    fig.add_vrect(x0=holdout_date, x1=price.index[-1], fillcolor=COLORS["accent"],
                  opacity=0.10, line_width=0, row=1, col=1,
                  annotation_text="Holdout (bloccato)", annotation_position="top left",
                  annotation_font_color=COLORS["text"])

    fig.add_trace(go.Scatter(
        x=osc.index, y=osc.values, name="Ciclo",
        line=dict(color=COLORS["accent"], width=1.2),
        hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.7,
                  line_dash="dash", row=2, col=1)

    fig.update_layout(
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        title=dict(text=f"{ticker} — Prezzo e ciclo isolato",
                   font=dict(size=16, color=COLORS["text"])),
        hovermode="x unified", legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333355", color=COLORS["text"])
    fig.update_yaxes(showgrid=True, gridcolor="#333355", color=COLORS["text"])
    return fig


# ============================================================
# 2. PERIODOGRAMMA CON SOGLIA DI SIGNIFICATIVITA'
# ============================================================
def build_spectrum_chart(periods: np.ndarray, power: np.ndarray,
                         threshold: np.ndarray | None, peaks: list,
                         dominant: float) -> go.Figure:
    """Periodogramma con soglia surrogata e picco dominante evidenziato."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=power, name="Potenza spettrale",
        line=dict(color=COLORS["primary"], width=1.6),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.10)",
        hovertemplate="Periodo %{x:.0f} barre<br>Potenza %{y:.3f}<extra></extra>"))

    if threshold is not None:
        fig.add_trace(go.Scatter(
            x=periods, y=threshold, name="Soglia significativita' (95° pct null)",
            line=dict(color=COLORS["negative"], width=1.2, dash="dash"),
            hovertemplate="Soglia %{y:.3f}<extra></extra>"))

    # NB Plotly: su un asse X logaritmico le coordinate di add_vline sono in unita' log10
    # (x=79 verrebbe posizionato a 10^79, facendo esplodere la scala). Passiamo log10(periodo).
    for pk in peaks:
        color = COLORS["positive"] if pk["significant"] else COLORS["neutral"]
        fig.add_vline(x=np.log10(pk["period"]), line_color=color, line_width=1,
                      line_dash="dot", opacity=0.6)

    if dominant == dominant:
        fig.add_vline(x=np.log10(dominant), line_color=COLORS["secondary"], line_width=2,
                      annotation_text=f"Dominante ~{dominant:.0f}",
                      annotation_font_color=COLORS["secondary"])

    fig.update_layout(**_base_layout(
        "Periodogramma (Lomb-Scargle) — dove sono i cicli",
        x_title="Periodo (barre)", y_title="Potenza normalizzata"))
    # Range esplicito (in unita' log10) per inquadrare esattamente la banda dei periodi
    # cercati ed evitare qualsiasi auto-range anomalo.
    if periods.size:
        lo = float(np.log10(periods.min())) - 0.03
        hi = float(np.log10(periods.max())) + 0.03
        fig.update_xaxes(type="log", range=[lo, hi])
    else:
        fig.update_xaxes(type="log")
    return fig


# ============================================================
# 3. SCALOGRAMMA (deriva del periodo nel tempo)
# ============================================================
def build_scalogram(mat: np.ndarray, periods: np.ndarray, times) -> go.Figure:
    """Heatmap tempo-periodo: mostra come il periodo dominante si sposta (non stazionarieta')."""
    fig = go.Figure(go.Heatmap(
        z=mat, x=times, y=periods, colorscale="Viridis",
        colorbar=dict(title="Ampiezza<br>(norm.)", tickfont=dict(color=COLORS["text"])),
        hovertemplate="Data %{x}<br>Periodo %{y:.0f}<br>Ampiezza %{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="Scalogramma — energia ciclica per periodo nel tempo",
                   font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(title="Data", color=COLORS["text"]),
        yaxis=dict(title="Periodo (barre)", color=COLORS["text"], type="log"),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ============================================================
# 4. OROLOGIO DI FASE (dove siamo nel ciclo, adesso)
# ============================================================
def build_phase_gauge(osc_value: float, state_label: str) -> go.Figure:
    """Gauge della posizione ciclica corrente in [-1, +1] con zone interpretative."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(np.clip(osc_value, -1, 1)),
        number=dict(font=dict(color=COLORS["text"], size=28),
                    valueformat=".2f"),
        title=dict(text=f"Posizione nel ciclo<br><span style='font-size:0.8em'>{state_label}</span>",
                   font=dict(color=COLORS["text"], size=15)),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor=COLORS["text"],
                      tickfont=dict(color=COLORS["text"])),
            bar=dict(color=COLORS["accent"], thickness=0.25),
            bgcolor=COLORS["surface"],
            steps=[
                dict(range=[-1, -0.4], color="rgba(76,175,80,0.35)"),   # zona minimo
                dict(range=[-0.4, 0.4], color="rgba(158,158,158,0.25)"),
                dict(range=[0.4, 1], color="rgba(244,67,54,0.35)"),     # zona massimo
            ],
            threshold=dict(line=dict(color=COLORS["text"], width=3),
                           thickness=0.75, value=float(np.clip(osc_value, -1, 1))),
        )))
    fig.update_layout(
        paper_bgcolor=COLORS["background"], font=dict(color=COLORS["text"]),
        margin=dict(l=30, r=30, t=60, b=10), height=280)
    return fig


# ============================================================
# 5. HURST ROLLING (regime)
# ============================================================
def build_hurst_chart(price: pd.Series, hurst: pd.Series) -> go.Figure:
    """Esponente di Hurst rolling con linea 0.5 e ombreggiatura dei regimi."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.06, subplot_titles=("Prezzo", "Esponente di Hurst (rolling)"))
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name="Prezzo",
                             line=dict(color=COLORS["primary"], width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hurst.index, y=hurst.values, name="Hurst",
                             line=dict(color=COLORS["secondary"], width=1.4),
                             hovertemplate="H=%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.5, line_color=COLORS["neutral"], line_width=1, line_dash="dash", row=2, col=1)
    fig.add_hrect(y0=0, y1=0.5, fillcolor=COLORS["positive"], opacity=0.07, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0.5, y1=1, fillcolor=COLORS["negative"], opacity=0.07, line_width=0, row=2, col=1)
    fig.update_layout(
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        title=dict(text="Regime: mean-reversion (H<0.5) vs trend (H>0.5)",
                   font=dict(size=16, color=COLORS["text"])),
        hovermode="x unified", legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=70, b=40))
    fig.update_xaxes(showgrid=True, gridcolor="#333355", color=COLORS["text"])
    fig.update_yaxes(showgrid=True, gridcolor="#333355", color=COLORS["text"])
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    return fig


# ============================================================
# 6. STAGIONALITA'
# ============================================================
def build_seasonality_bars(stats_df: pd.DataFrame, title: str) -> go.Figure:
    """Barre del rendimento medio per bucket; colore = significativita'/segno."""
    colors = []
    for _, row in stats_df.iterrows():
        if row["significant"] and row["mean_pct"] > 0:
            colors.append(COLORS["positive"])
        elif row["significant"] and row["mean_pct"] < 0:
            colors.append(COLORS["negative"])
        else:
            colors.append(COLORS["neutral"])

    hover = [f"Media {r['mean_pct']:+.3f}%<br>p-value {r['p_value']:.3f}<br>n={int(r['n'])}"
             for _, r in stats_df.iterrows()]

    fig = go.Figure(go.Bar(
        x=list(stats_df.index), y=stats_df["mean_pct"], marker_color=colors,
        text=[("★" if s else "") for s in stats_df["significant"]],
        textposition="outside", hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8)
    fig.update_layout(**_base_layout(title, x_title="", y_title="Rendimento medio (%)"))
    fig.update_layout(hovermode="closest")
    return fig


# ============================================================
# 7. EQUITY CURVE (strategia vs buy & hold) con confine IS/OOS
# ============================================================
def build_equity_chart(equity_strat: pd.Series, equity_bh: pd.Series,
                       split_date) -> go.Figure:
    """Equity base 100 strategia vs buy&hold, con marcatore del confine IS/OOS."""
    es = equity_strat / equity_strat.iloc[0] * 100
    eb = equity_bh / equity_bh.iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=es.index, y=es.values, name="Strategia ciclica",
                             line=dict(color=COLORS["primary"], width=2),
                             fill="tozeroy", fillcolor="rgba(33,150,243,0.08)"))
    fig.add_trace(go.Scatter(x=eb.index, y=eb.values, name="Buy & Hold",
                             line=dict(color=COLORS["neutral"], width=1.4, dash="dot")))
    fig.add_vline(x=split_date, line_color=COLORS["secondary"], line_width=1.5,
                  annotation_text="inizio OOS", annotation_font_color=COLORS["secondary"])
    fig.add_hline(y=100, line_color=COLORS["neutral"], line_width=0.7, line_dash="dash", opacity=0.5)
    fig.update_layout(**_base_layout("Equity Curve (base 100) — Strategia vs Buy & Hold",
                                     x_title="Data", y_title="NAV (base 100)"))
    return fig


# ============================================================
# 7b. EQUITY SUL SOLO HOLDOUT (esame finale)
# ============================================================
def build_holdout_equity(equity_strat: pd.Series, equity_bh: pd.Series) -> go.Figure:
    """Equity base 100 sul solo segmento holdout: strategia congelata vs buy & hold."""
    es = equity_strat / equity_strat.iloc[0] * 100
    eb = equity_bh / equity_bh.iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=es.index, y=es.values, name="Strategia congelata",
                             line=dict(color=COLORS["accent"], width=2),
                             fill="tozeroy", fillcolor="rgba(171,71,188,0.08)"))
    fig.add_trace(go.Scatter(x=eb.index, y=eb.values, name="Buy & Hold",
                             line=dict(color=COLORS["neutral"], width=1.4, dash="dot")))
    fig.add_hline(y=100, line_color=COLORS["neutral"], line_width=0.7,
                  line_dash="dash", opacity=0.5)
    fig.update_layout(**_base_layout("Esame finale — equity sul solo holdout (base 100)",
                                     x_title="Data", y_title="NAV (base 100)"))
    return fig


# ============================================================
# 8. DRAWDOWN
# ============================================================
def build_drawdown_chart(equity: pd.Series) -> go.Figure:
    """Drawdown dal massimo storico."""
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0) * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy",
        fillcolor="rgba(244,67,54,0.3)", line=dict(color=COLORS["negative"], width=1.3),
        hovertemplate="%{y:.1f}%<extra></extra>"))
    fig.update_layout(**_base_layout("Drawdown della strategia", x_title="Data",
                                     y_title="Drawdown (%)"))
    return fig


# ============================================================
# 9. DISTRIBUZIONE NULLA (test di significativita')
# ============================================================
def build_null_distribution(null_values: np.ndarray, strat_value: float,
                            p_value: float, metric_name: str = "Rendimento OOS") -> go.Figure:
    """Istogramma del null random-entry con la strategia reale evidenziata."""
    fig = go.Figure()
    if null_values.size:
        fig.add_trace(go.Histogram(
            x=null_values * 100, nbinsx=40, name="Strategie random-entry (null)",
            marker_color=COLORS["neutral"], opacity=0.75,
            histnorm="probability density"))
    fig.add_vline(x=strat_value * 100, line_color=COLORS["positive"], line_width=3,
                  annotation_text=f"Strategia ({strat_value:+.1%})",
                  annotation_font_color=COLORS["positive"])
    if null_values.size:
        p75 = np.percentile(null_values, 75) * 100
        fig.add_vline(x=p75, line_color=COLORS["secondary"], line_width=1.2, line_dash="dash",
                      annotation_text="75° pct null", annotation_font_color=COLORS["secondary"])
    fig.update_layout(**_base_layout(
        f"Test di significativita' — {metric_name} vs ipotesi nulla (p={p_value:.2f})",
        x_title=f"{metric_name} (%)", y_title="Densita'"))
    fig.update_layout(hovermode="closest")
    return fig


# ============================================================
# 10. WALK-FORWARD (rendimenti OOS per fold)
# ============================================================
def build_walkforward_bars(folds: list) -> go.Figure:
    """Barre del rendimento OOS per ogni fold del walk-forward."""
    labels = [f"Fold {f['fold']}<br>{pd.Timestamp(f['test_start']).date()}" for f in folds]
    vals = [f["total_return"] * 100 for f in folds]
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in vals]
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
        hovertext=[f"Sharpe {f['sharpe']:.2f}<br>MaxDD {f['max_dd']:.1%}" for f in folds],
        hoverinfo="text"))
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8)
    fig.update_layout(**_base_layout("Walk-Forward — rendimento OOS per fold",
                                     x_title="", y_title="Rendimento OOS (%)"))
    fig.update_layout(hovermode="closest")
    return fig
