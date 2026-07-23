"""
cycles.py — Analisi spettrale e misura del ciclo dominante.

Contenuto:
    1. Periodogrammi (Welch e Lomb-Scargle) su serie detrendizzate.
    2. Significativita' spettrale via surrogati a fase randomizzata (evita di
       scambiare picchi casuali per cicli reali: un periodogramma trova SEMPRE picchi).
    3. Stima del periodo dominante (per la calibrazione IN-SAMPLE).
    4. Oscillatore ciclico causale + fase istantanea (per i segnali, senza look-ahead).
    5. Banco di passa-banda per lo scalogramma (mostra la deriva del periodo nel tempo).

Filosofia anti-overfitting: il periodo dominante si stima UNA VOLTA sull'in-sample e si
"congela"; l'out-of-sample usa quel periodo senza sbirciare nel futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from src.dsp import high_pass, roofing_filter, band_pass, agc, super_smoother


# ============================================================
# SURROGATI A FASE RANDOMIZZATA (ipotesi nulla spettrale)
# ============================================================
def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Genera un surrogato a fase randomizzata: conserva lo spettro di ampiezza (quindi
    l'autocorrelazione lineare) ma distrugge la struttura di fase deterministica.
    E' l'ipotesi nulla corretta per testare se un picco spettrale e' "vero".

    Args:
        x:   serie detrendizzata (media ~0)
        rng: generatore numpy

    Returns:
        ndarray surrogato, stessa lunghezza e stesso spettro di ampiezza di x.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    # Fasi casuali; DC e Nyquist restano reali
    random_phases = rng.uniform(0, 2 * np.pi, size=mag.size)
    random_phases[0] = 0.0
    if n % 2 == 0:
        random_phases[-1] = 0.0
    surrogate = np.fft.irfft(mag * np.exp(1j * random_phases), n=n)
    return surrogate


# ============================================================
# PERIODOGRAMMI
# ============================================================
def _period_grid(min_period: float, max_period: float, n: int = 400) -> np.ndarray:
    """Griglia di periodi spaziata log tra min e max."""
    return np.logspace(np.log10(min_period), np.log10(max_period), n)


def lombscargle_periodogram(x, min_period: float = 5, max_period: float = 250,
                            n_periods: int = 400):
    """
    Periodogramma di Lomb-Scargle. Robusto ai gap (weekend/festivita') perche' non
    richiede campionamento uniforme: adatto ai prezzi giornalieri.

    Args:
        x:          serie DETRENDIZZATA (array-like, media ~0)
        min_period: periodo minimo da cercare (barre)
        max_period: periodo massimo da cercare (barre)
        n_periods:  risoluzione della griglia

    Returns:
        (periods, power): array dei periodi e potenza normalizzata associata.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    t = np.arange(x.size, dtype=float)
    periods = _period_grid(min_period, max_period, n_periods)
    ang_freqs = 2.0 * np.pi / periods
    power = sp_signal.lombscargle(t, x, ang_freqs, normalize=True)
    return periods, power


def welch_periodogram(x, min_period: float = 5, max_period: float = 250,
                      nperseg: int | None = None):
    """
    Densita' spettrale di potenza con il metodo di Welch (media di periodogrammi su
    segmenti sovrapposti -> stima piu' liscia e stabile).

    Args:
        x:          serie DETRENDIZZATA (array-like)
        min_period: periodo minimo di interesse
        max_period: periodo massimo di interesse
        nperseg:    lunghezza segmento (default: min(len//2, 512))

    Returns:
        (periods, power) filtrati sul range [min_period, max_period].
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    if nperseg is None:
        nperseg = int(min(x.size // 2, 512))
    nperseg = max(nperseg, 8)
    freqs, psd = sp_signal.welch(x, fs=1.0, nperseg=nperseg, detrend="linear")
    # Evita divisione per zero alla frequenza 0
    valid = freqs > 0
    periods = 1.0 / freqs[valid]
    power = psd[valid]
    mask = (periods >= min_period) & (periods <= max_period)
    order = np.argsort(periods[mask])
    return periods[mask][order], power[mask][order]


def spectral_significance(x, min_period: float = 5, max_period: float = 250,
                          n_periods: int = 400, n_surrogates: int = 200,
                          percentile: float = 95.0, seed: int = 42):
    """
    Soglia di significativita' del periodogramma di Lomb-Scargle via surrogati a fase
    randomizzata. Un picco che supera la soglia e' improbabile sotto l'ipotesi nulla.

    Args:
        x:            serie detrendizzata
        min_period:   periodo minimo
        max_period:   periodo massimo
        n_periods:    risoluzione griglia
        n_surrogates: numero di surrogati (200 = buon compromesso)
        percentile:   percentile della distribuzione nulla usato come soglia (95 = ~p0.05)
        seed:         seme per riproducibilita'

    Returns:
        (periods, power, threshold): la soglia e' una CURVA per periodo.
    """
    periods, power = lombscargle_periodogram(x, min_period, max_period, n_periods)
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=float).ravel()
    x_arr = x_arr - x_arr.mean()

    null_powers = np.empty((n_surrogates, periods.size))
    for k in range(n_surrogates):
        surr = phase_randomize(x_arr, rng)
        _, p = lombscargle_periodogram(surr, min_period, max_period, n_periods)
        null_powers[k] = p
    threshold = np.percentile(null_powers, percentile, axis=0)
    return periods, power, threshold


def dominant_period(x, min_period: float = 10, max_period: float = 120,
                    method: str = "lombscargle") -> float:
    """
    Periodo dominante: il picco piu' alto del periodogramma nel range consentito.
    Da calcolare SOLO sull'in-sample per congelare l'iperparametro.

    Args:
        x:          serie detrendizzata
        min_period: limite inferiore ricerca
        max_period: limite superiore ricerca
        method:     'lombscargle' (default) o 'welch'

    Returns:
        Periodo dominante in barre (float). NaN se non stimabile.
    """
    if method == "welch":
        periods, power = welch_periodogram(x, min_period, max_period)
    else:
        periods, power = lombscargle_periodogram(x, min_period, max_period)
    if power.size == 0:
        return np.nan
    return float(periods[np.argmax(power)])


def find_peaks(periods: np.ndarray, power: np.ndarray,
               threshold: np.ndarray | None = None, top_k: int = 5):
    """
    Estrae i picchi locali del periodogramma, ordinati per potenza. Se e' fornita una
    soglia, tiene solo i picchi che la superano (significativi).

    Returns:
        Lista di dict [{period, power, significant}], ordinata per potenza decrescente.
    """
    peak_idx, _ = sp_signal.find_peaks(power)
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(power))]) if power.size else np.array([])
    peaks = []
    for i in peak_idx:
        sig = bool(threshold is not None and power[i] > threshold[i])
        peaks.append({"period": float(periods[i]), "power": float(power[i]),
                      "significant": sig})
    peaks.sort(key=lambda d: d["power"], reverse=True)
    return peaks[:top_k]


# ============================================================
# OSCILLATORE CICLICO CAUSALE + FASE ISTANTANEA (per i segnali)
# ============================================================
def cycle_oscillator(price: pd.Series, period: float, bandwidth: float = 0.3) -> pd.Series:
    """
    Oscillatore ciclico normalizzato in ~[-1, 1], centrato sul periodo dato. Causale:
    usa un passa-banda ricorsivo di Ehlers + AGC. Adatto ai segnali (niente look-ahead).

    Args:
        price:     Serie di prezzi (index datetime)
        period:    periodo del ciclo da isolare
        bandwidth: larghezza di banda relativa del filtro

    Returns:
        pd.Series dell'oscillatore, allineata a `price`.
    """
    log_price = np.log(np.clip(price.values, 1e-12, None))
    bp, _ = band_pass(log_price, period, bandwidth)
    osc = agc(bp)
    return pd.Series(osc, index=price.index, name="cycle_osc")


def instantaneous_phase(osc: pd.Series, period: float) -> pd.Series:
    """
    Fase istantanea (radianti) dell'oscillatore, calcolata in modo CAUSALE.

    La quadratura (componente a 90 gradi) e' approssimata dalla derivata normalizzata:
        per sin(w t), d/dt = w cos(w t); dividendo per w = 2*pi/period si ottiene la
        componente coseno con la stessa ampiezza -> fase = atan2(quadratura, oscillatore).

    Args:
        osc:    oscillatore ciclico (output di cycle_oscillator)
        period: periodo del ciclo (per la scala della quadratura)

    Returns:
        pd.Series della fase in radianti in [-pi, pi].
    """
    o = osc.values
    n = o.size
    k = period / (2.0 * np.pi)
    phase = np.zeros(n)
    for i in range(1, n):
        quad = k * (o[i] - o[i - 1])
        phase[i] = np.arctan2(quad, o[i])
    return pd.Series(phase, index=osc.index, name="phase")


def bandpass_bank(price: pd.Series, periods: np.ndarray, bandwidth: float = 0.3) -> np.ndarray:
    """
    Banco di filtri passa-banda: ampiezza causale a ogni periodo, per ogni barra.
    Base per lo scalogramma (heatmap tempo-periodo) che mostra come il ciclo dominante
    si sposta nel tempo (non stazionarieta').

    Args:
        price:     Serie di prezzi
        periods:   array di periodi da valutare
        bandwidth: larghezza di banda relativa

    Returns:
        ndarray 2D shape (len(periods), len(price)) con l'ampiezza normalizzata [0,1]
        per colonna (ogni istante temporale).
    """
    log_price = np.log(np.clip(price.values, 1e-12, None))
    mat = np.zeros((periods.size, log_price.size))
    for j, p in enumerate(periods):
        _, amp = band_pass(log_price, p, bandwidth)
        mat[j] = amp
    # Normalizza per colonna (per istante) cosi' si vede il periodo dominante nel tempo
    col_max = mat.max(axis=0, keepdims=True)
    col_max[col_max < 1e-12] = 1.0
    return mat / col_max
