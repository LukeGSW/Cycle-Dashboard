"""
hurst.py — Esponente di Hurst (H. E. Hurst) come misuratore di REGIME.

ATTENZIONE alla omonimia: qui parliamo dell'esponente di Hurst (statistica di memoria
lunga), NON dei cicli di J. M. Hurst (teoria di trading). Sono due persone diverse.

Interpretazione dell'esponente H:
    H < 0.5  -> serie ANTI-persistente / mean-reverting (favorevole al cycle-timing)
    H = 0.5  -> random walk (nessuna memoria)
    H > 0.5  -> serie PERSISTENTE / trending (il cycle-timing e' meno affidabile)

Due stimatori:
    - R/S analysis (Rescaled Range): classico, storico.
    - DFA (Detrended Fluctuation Analysis): piu' robusto su serie finanziarie non
      stazionarie, e' il default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# R/S ANALYSIS (Rescaled Range)
# ============================================================
def hurst_rs(x, min_chunk: int = 8) -> float:
    """
    Stima l'esponente di Hurst con l'analisi Rescaled Range (R/S).

    Args:
        x:         serie (tipicamente rendimenti log), array-like
        min_chunk: dimensione minima dei sotto-blocchi

    Returns:
        Esponente di Hurst (float). NaN se dati insufficienti.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2 * min_chunk:
        return np.nan

    # Dimensioni dei blocchi in scala logaritmica
    max_chunk = n // 2
    sizes = np.unique(np.floor(np.logspace(
        np.log10(min_chunk), np.log10(max_chunk), 12)).astype(int))
    sizes = sizes[sizes >= min_chunk]

    rs_means = []
    valid_sizes = []
    for s in sizes:
        n_chunks = n // s
        if n_chunks < 1:
            continue
        rs_vals = []
        for k in range(n_chunks):
            chunk = x[k * s:(k + 1) * s]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            R = dev.max() - dev.min()
            S = chunk.std()
            if S > 1e-12:
                rs_vals.append(R / S)
        if rs_vals:
            rs_means.append(np.mean(rs_vals))
            valid_sizes.append(s)

    if len(valid_sizes) < 3:
        return np.nan

    # Pendenza di log(R/S) vs log(size) = esponente di Hurst
    slope = np.polyfit(np.log(valid_sizes), np.log(rs_means), 1)[0]
    return float(slope)


# ============================================================
# DFA (Detrended Fluctuation Analysis)
# ============================================================
def hurst_dfa(x, min_scale: int = 8, order: int = 1) -> float:
    """
    Stima l'esponente di Hurst con la DFA. Piu' robusto della R/S su serie finanziarie.

    Args:
        x:         serie (tipicamente rendimenti log), array-like
        min_scale: scala minima (in barre)
        order:     ordine del polinomio di detrend per segmento (1 = lineare)

    Returns:
        Esponente di Hurst / DFA (float). NaN se dati insufficienti.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4 * min_scale:
        return np.nan

    # Profilo integrato (cumulata degli scarti dalla media)
    y = np.cumsum(x - x.mean())

    max_scale = n // 4
    scales = np.unique(np.floor(np.logspace(
        np.log10(min_scale), np.log10(max_scale), 12)).astype(int))
    scales = scales[scales >= min_scale]

    fluct = []
    valid_scales = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        rms = []
        # Segmenti dall'inizio e dalla fine (uso completo dei dati)
        for start in (0, n - n_seg * s):
            for k in range(n_seg):
                seg = y[start + k * s: start + (k + 1) * s]
                t = np.arange(s)
                coeffs = np.polyfit(t, seg, order)
                trend = np.polyval(coeffs, t)
                rms.append(np.mean((seg - trend) ** 2))
        if rms:
            fluct.append(np.sqrt(np.mean(rms)))
            valid_scales.append(s)

    if len(valid_scales) < 3:
        return np.nan

    slope = np.polyfit(np.log(valid_scales), np.log(fluct), 1)[0]
    return float(slope)


# ============================================================
# ROLLING HURST — regime nel tempo (causale)
# ============================================================
def rolling_hurst(price: pd.Series, window: int = 200, method: str = "dfa") -> pd.Series:
    """
    Esponente di Hurst su finestra mobile, calcolato sui rendimenti log. Causale:
    ogni valore usa solo la finestra di dati passata.

    Args:
        price:  Serie di prezzi (index datetime)
        window: ampiezza della finestra mobile (barre). >=150 consigliato.
        method: 'dfa' (default) o 'rs'

    Returns:
        pd.Series dell'esponente di Hurst, allineata a `price` (NaN nella fase iniziale).
    """
    log_ret = np.log(price).diff()
    estimator = hurst_dfa if method == "dfa" else hurst_rs

    values = np.full(len(price), np.nan)
    arr = log_ret.values
    for i in range(window, len(price)):
        values[i] = estimator(arr[i - window + 1:i + 1])
    return pd.Series(values, index=price.index, name=f"hurst_{method}")
