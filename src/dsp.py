"""
dsp.py — Filtri digitali causali per l'analisi ciclica (stile John Ehlers).

Perche' Ehlers e non la trasformata di Hilbert di scipy?
    scipy.signal.hilbert lavora con una FFT sull'intera serie: il valore al tempo t
    dipende da dati FUTURI. In un backtest questo introduce LOOK-AHEAD BIAS.
    Tutti i filtri qui sono RICORSIVI e usano solo il passato -> completamente causali,
    quindi utilizzabili per generare segnali onesti.

Tutte le funzioni accettano un array-like (Series o ndarray) e restituiscono un ndarray
float della stessa lunghezza. La gestione dell'indice pandas avviene nei moduli chiamanti.

Riferimenti:
    J. Ehlers, "Cycle Analytics for Traders" (2013) e "Cybernetic Analysis" (2004).
"""

from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)


def _as_array(x) -> np.ndarray:
    """Converte input array-like in ndarray float 1D contiguo."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr


# ============================================================
# SUPER SMOOTHER (passa-basso a 2 poli, Ehlers)
# ============================================================
def super_smoother(x, period: float) -> np.ndarray:
    """
    Filtro Super Smoother di Ehlers: rimuove le componenti a periodo piu' corto di
    `period` (rumore ad alta frequenza) con ritardo minimo e senza aliasing.

    Args:
        x:      serie di input (array-like)
        period: periodo di taglio (in barre). Componenti piu' veloci vengono attenuate.

    Returns:
        ndarray filtrato, stessa lunghezza di x.
    """
    x = _as_array(x)
    n = x.size
    y = np.copy(x)
    if n < 3 or period <= 1:
        return y

    a1 = np.exp(-SQRT2 * np.pi / period)
    b1 = 2.0 * a1 * np.cos(SQRT2 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    for i in range(2, n):
        y[i] = c1 * (x[i] + x[i - 1]) / 2.0 + c2 * y[i - 1] + c3 * y[i - 2]
    return y


# ============================================================
# HIGH-PASS a 2 poli (Ehlers) — rimuove il trend
# ============================================================
def high_pass(x, period: float) -> np.ndarray:
    """
    Filtro passa-alto a 2 poli di Ehlers: rimuove le componenti a periodo piu' LUNGO
    di `period` (il trend e i cicli lentissimi), lasciando i cicli piu' rapidi.

    Args:
        x:      serie di input (array-like)
        period: periodo di taglio; cio' che e' piu' lento viene rimosso.

    Returns:
        ndarray filtrato (serie detrendizzata), media ~0.
    """
    x = _as_array(x)
    n = x.size
    y = np.zeros(n)
    if n < 3 or period <= 1:
        return y

    # alpha del passa-alto a 2 poli
    arg = SQRT2 * np.pi / period
    alpha = (np.cos(arg) + np.sin(arg) - 1.0) / np.cos(arg)
    c = (1.0 - alpha / 2.0) ** 2
    one_minus = 1.0 - alpha

    for i in range(2, n):
        y[i] = (
            c * (x[i] - 2.0 * x[i - 1] + x[i - 2])
            + 2.0 * one_minus * y[i - 1]
            - one_minus * one_minus * y[i - 2]
        )
    return y


# ============================================================
# ROOFING FILTER (Ehlers) = high-pass + super smoother
# ============================================================
def roofing_filter(x, hp_period: float = 125.0, ss_period: float = 10.0) -> np.ndarray:
    """
    Roofing filter di Ehlers: passa-banda che tiene solo i cicli con periodo compreso
    tra `ss_period` e `hp_period`. Rimuove sia il trend (frequenze basse) sia il
    rumore (frequenze alte). E' la serie ideale su cui misurare i cicli.

    Args:
        x:         serie di input (tipicamente log-prezzo)
        hp_period: taglio del passa-alto: cicli piu' lunghi di questo = trend, rimossi.
        ss_period: taglio del passa-basso: cicli piu' corti di questo = rumore, rimossi.

    Returns:
        ndarray filtrato (oscillatore ciclico grezzo, media ~0).
    """
    hp = high_pass(x, hp_period)
    return super_smoother(hp, ss_period)


# ============================================================
# BAND-PASS (Ehlers) — isola un singolo ciclo
# ============================================================
def band_pass(x, period: float, bandwidth: float = 0.3):
    """
    Filtro passa-banda di Ehlers centrato su `period`. Isola la componente ciclica
    con quel periodo; la larghezza relativa e' controllata da `bandwidth`.

    Args:
        x:         serie di input (array-like)
        period:    periodo centrale del filtro (in barre)
        bandwidth: larghezza di banda relativa (0.2-0.4 tipico). Piu' piccola = piu'
                   selettiva ma piu' ritardo.

    Returns:
        (bp, amp): bp = output del passa-banda (oscilla intorno a 0);
                   amp = inviluppo di ampiezza causale (radice della potenza lisciata).
    """
    x = _as_array(x)
    n = x.size
    bp = np.zeros(n)
    if n < 3 or period <= 1:
        return bp, np.zeros(n)

    beta = np.cos(2.0 * np.pi / period)
    gamma = 1.0 / np.cos(4.0 * np.pi * bandwidth / period)
    alpha = gamma - np.sqrt(max(gamma * gamma - 1.0, 0.0))

    for i in range(2, n):
        bp[i] = (
            0.5 * (1.0 - alpha) * (x[i] - x[i - 2])
            + beta * (1.0 + alpha) * bp[i - 1]
            - alpha * bp[i - 2]
        )

    # Inviluppo di ampiezza causale: sqrt della potenza lisciata dal super smoother.
    power = super_smoother(bp ** 2, period)
    amp = np.sqrt(np.clip(power, 0.0, None))
    return bp, amp


# ============================================================
# AGC — Automatic Gain Control (normalizza in ~[-1, 1])
# ============================================================
def agc(bp, decay: float = 0.991) -> np.ndarray:
    """
    Automatic Gain Control di Ehlers: normalizza un oscillatore in ~[-1, 1] dividendo
    per un picco che decade lentamente. Causale.

    Args:
        bp:    output di un passa-banda (array-like)
        decay: fattore di decadimento del picco (0.99 tipico).

    Returns:
        ndarray normalizzato ~[-1, 1].
    """
    bp = _as_array(bp)
    n = bp.size
    peak = np.zeros(n)
    out = np.zeros(n)
    for i in range(1, n):
        peak[i] = max(decay * peak[i - 1], abs(bp[i]))
        out[i] = bp[i] / peak[i] if peak[i] > 1e-12 else 0.0
    return out


# ============================================================
# DETREND — wrapper per l'analisi spettrale
# ============================================================
def detrend(x, method: str = "highpass", hp_period: float = 200.0,
            window: int = 100) -> np.ndarray:
    """
    Rimuove il trend prima dell'analisi spettrale (il trend domina lo spettro e
    maschera i cicli). Diversi metodi a seconda dell'esigenza.

    Args:
        x:         serie di input (array-like)
        method:    'highpass'  -> passa-alto Ehlers (default, morbido)
                   'logdiff'   -> differenza dei log (rendimenti)
                   'diff'      -> differenza prima
                   'zscore'    -> (x - media mobile) / std mobile
        hp_period: cutoff per method='highpass'
        window:    finestra per method='zscore'

    Returns:
        ndarray detrendizzato.
    """
    x = _as_array(x)
    if method == "highpass":
        return high_pass(x, hp_period)
    if method == "logdiff":
        lx = np.log(np.clip(x, 1e-12, None))
        d = np.diff(lx, prepend=lx[0])
        return d
    if method == "diff":
        return np.diff(x, prepend=x[0])
    if method == "zscore":
        s = np.full_like(x, np.nan)
        for i in range(window, x.size):
            w = x[i - window:i]
            mu, sd = w.mean(), w.std()
            s[i] = (x[i] - mu) / sd if sd > 1e-12 else 0.0
        return np.nan_to_num(s)
    raise ValueError(f"Metodo di detrend sconosciuto: {method}")
