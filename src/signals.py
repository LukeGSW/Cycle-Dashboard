"""
signals.py — Generazione dei segnali ciclici + confluenza.

Segnale base: SINEWAVE di Ehlers. Dalla fase istantanea causale costruiamo:
    sine      = sin(fase)
    lead_sine = sin(fase + 45 gradi)
Un incrocio di `sine` sopra `lead_sine` segnala il minimo del ciclo (giro rialzista);
l'incrocio opposto segnala il massimo (giro ribassista). E' il modo canonico e oggettivo
per marcare le svolte del ciclo, molto meno soggetto a ottimizzazione di un semplice
crossover di medie.

CONVENZIONE: le posizioni qui NON sono shiftate. Sono la posizione target a chiusura di t.
Lo shift +1 (esecuzione a t+1) e' applicato dentro backtest.run_backtest -> niente look-ahead.

Confluenza: la teoria dice che il cycle-timing funziona meglio in regime mean-reverting
(Hurst basso). La maschera di regime azzera le posizioni quando il mercato e' in forte
trend (Hurst alto), dove il timing dei cicli e' meno affidabile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cycles import cycle_oscillator, instantaneous_phase
from src.hurst import rolling_hurst


def cycle_position(price: pd.Series, period: float, bandwidth: float = 0.3,
                   mode: str = "long_flat") -> pd.Series:
    """
    Posizione ciclica dal crossover sinewave di Ehlers.

    Args:
        price:     Serie di prezzi
        period:    periodo del ciclo (dall'analisi spettrale in-sample)
        bandwidth: larghezza di banda del passa-banda
        mode:      'long_flat'  -> {0, 1}  (lungo nella fase di salita, flat nella discesa)
                   'long_short' -> {-1, 1} (lungo/corto)

    Returns:
        pd.Series della posizione target (NON shiftata), allineata a `price`.
    """
    osc = cycle_oscillator(price, period, bandwidth)
    phase = instantaneous_phase(osc, period)

    sine = np.sin(phase.values)
    lead = np.sin(phase.values + np.pi / 4.0)

    cross_up = np.zeros(len(price), dtype=bool)
    cross_dn = np.zeros(len(price), dtype=bool)
    cross_up[1:] = (sine[:-1] <= lead[:-1]) & (sine[1:] > lead[1:])
    cross_dn[1:] = (sine[:-1] >= lead[:-1]) & (sine[1:] < lead[1:])

    long_val = 1.0
    flat_val = 0.0 if mode == "long_flat" else -1.0

    state = np.full(len(price), np.nan)
    state[cross_up] = long_val
    state[cross_dn] = flat_val
    pos = pd.Series(state, index=price.index).ffill().fillna(0.0)
    pos.name = "cycle_position"
    return pos


def regime_mask(price: pd.Series, window: int = 200, method: str = "dfa",
                max_hurst: float = 0.55) -> pd.Series:
    """
    Maschera di regime favorevole al cycle-timing (Hurst basso = mean-reverting).

    Args:
        price:     Serie di prezzi
        window:    finestra dell'esponente di Hurst rolling
        method:    'dfa' o 'rs'
        max_hurst: soglia sopra la quale il regime e' 'troppo trending' -> maschera 0

    Returns:
        pd.Series booleana (1.0 dove il regime e' favorevole, 0.0 altrimenti). Dove
        l'Hurst non e' ancora calcolabile (fase iniziale) la maschera vale 1.0
        (non filtra), per non perdere dati.
    """
    h = rolling_hurst(price, window=window, method=method)
    mask = (h <= max_hurst).astype(float)
    mask[h.isna()] = 1.0
    mask.name = "regime_mask"
    return mask


def seasonal_mask(price: pd.Series, favorable_months: list[int]) -> pd.Series:
    """
    Maschera stagionale: 1.0 nei mesi con bias rialzista significativo (misurati
    in-sample), 0.0 altrove. Se `favorable_months` e' vuota, non filtra (tutti 1.0).

    Args:
        price:            Serie di prezzi (serve solo per l'indice)
        favorable_months: lista di mesi 0=Gen..11=Dic considerati favorevoli

    Returns:
        pd.Series booleana allineata a price.
    """
    if not favorable_months:
        return pd.Series(1.0, index=price.index, name="seasonal_mask")
    months = price.index.month - 1
    mask = pd.Series(np.isin(months, favorable_months).astype(float),
                     index=price.index, name="seasonal_mask")
    return mask


def apply_confluence(position: pd.Series, masks: list[pd.Series]) -> pd.Series:
    """
    Applica la confluenza: la posizione sopravvive solo dove TUTTE le maschere sono
    attive (AND logico). Le maschere devono essere ~indipendenti dal segnale ciclico
    per aggiungere informazione (spettro + regime + stagionalita').

    Args:
        position: posizione ciclica base
        masks:    lista di Serie booleane (1.0/0.0) allineate a position

    Returns:
        pd.Series della posizione filtrata (NON shiftata).
    """
    out = position.copy()
    for m in masks:
        out = out * m.reindex(out.index).fillna(0.0)
    out.name = "confluence_position"
    return out
