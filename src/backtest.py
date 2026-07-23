"""
backtest.py — Backtest vettoriale CAUSALE e metriche di performance.

CONVENZIONE ANTI-LOOK-AHEAD (unica fonte di verita'):
    - I moduli di segnale restituiscono la posizione TARGET decisa alla CHIUSURA della
      barra t (usando solo dati fino a t incluso). NON e' pre-shiftata.
    - Qui applichiamo position.shift(1): l'esposizione decisa a chiusura di t guadagna
      il rendimento da t a t+1. Impossibile "vedere" il futuro.
    - I costi si pagano quando la posizione (gia' shiftata) cambia.

Tutte le metriche assumono barre giornaliere (252/anno) salvo diverso `periods_per_year`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def pct_returns(price: pd.Series) -> pd.Series:
    """Rendimenti semplici barra-su-barra."""
    return price.pct_change().fillna(0.0)


def buy_and_hold_returns(price: pd.Series) -> pd.Series:
    """Rendimenti del benchmark buy & hold (sempre lungo)."""
    return pct_returns(price)


def count_trades(position: pd.Series) -> int:
    """
    Numero di 'trade' = numero di variazioni di posizione (ingressi + uscite + inversioni).
    Un ciclo completo long->flat conta 2; long->short conta 2.
    """
    pos = position.fillna(0.0)
    changes = (pos != pos.shift(1)).sum()
    return int(changes)


def count_entries(position: pd.Series) -> int:
    """Numero di INGRESSI (transizioni da 0/negativo a positivo, o da 0 a diverso da 0)."""
    pos = position.fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    entries = ((prev == 0) & (pos != 0)).sum()
    return int(entries)


def run_backtest(price: pd.Series, position: pd.Series,
                 cost_bps: float = 2.0) -> dict:
    """
    Esegue il backtest causale della strategia.

    Args:
        price:    Serie di prezzi (index datetime)
        position: posizione target decisa a chiusura di t (NON shiftata). Valori
                  tipici: 0/1 (long-flat) oppure -1/0/+1 (long-short).
        cost_bps: costo di transazione in basis points per cambio di 1.0 di esposizione
                  (round-turn ~ 2*cost su un ciclo entra/esci). Include commissioni+slippage.

    Returns:
        dict con:
            'strat_returns'  Serie rendimenti netti della strategia
            'equity'         Serie equity (base 1.0)
            'exposure_used'  posizione effettivamente in vigore (shiftata di 1)
            'gross_returns'  rendimenti lordi (senza costi)
            'turnover'       Serie del turnover per barra
            'n_trades'       numero variazioni di posizione
    """
    ret = pct_returns(price)
    exposure = position.shift(1).fillna(0.0)            # decisa a t, attiva da t+1
    gross = exposure * ret

    # Costi: proporzionali alla variazione di esposizione (in frazione di capitale)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    return {
        "strat_returns": net,
        "gross_returns": gross,
        "equity": equity,
        "exposure_used": exposure,
        "turnover": turnover,
        "n_trades": count_trades(position),
    }


def equity_curve(returns: pd.Series) -> pd.Series:
    """Equity (base 1.0) da una serie di rendimenti."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    """Massimo drawdown (frazione negativa, es. -0.23 = -23%)."""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def compute_metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """
    Metriche di performance risk-adjusted da una serie di rendimenti.

    Returns:
        dict: total_return, cagr, ann_vol, sharpe, sortino, max_dd, mar, hit_rate,
              n_periods. Tutti in frazione (non %), tranne i conteggi.
    """
    r = returns.fillna(0.0)
    n = len(r)
    if n == 0:
        return {k: np.nan for k in
                ["total_return", "cagr", "ann_vol", "sharpe", "sortino",
                 "max_dd", "mar", "hit_rate", "n_periods"]}

    eq = equity_curve(r)
    total_return = float(eq.iloc[-1] - 1.0)
    years = n / periods_per_year
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else np.nan

    mean = r.mean()
    std = r.std(ddof=1)
    ann_vol = float(std * np.sqrt(periods_per_year)) if std == std else np.nan
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 1e-12 else 0.0

    downside = r[r < 0].std(ddof=1)
    sortino = float(mean / downside * np.sqrt(periods_per_year)) if downside > 1e-12 else np.nan

    mdd = max_drawdown(eq)
    mar = float(cagr / abs(mdd)) if mdd < -1e-9 and cagr == cagr else np.nan

    active = r[r != 0.0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": mdd,
        "mar": mar,
        "hit_rate": hit_rate,
        "n_periods": n,
    }
