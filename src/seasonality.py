"""
seasonality.py — Effetti stagionali/calendariali con test di significativita'.

E' il sottoinsieme TESTABILE del folklore ciclico (a differenza di Gann/Elliott):
giorno della settimana, mese dell'anno, giorno del mese. Ogni effetto e' corredato da
un t-test contro l'ipotesi nulla "media = 0" cosi' l'utente non scambia rumore per
segnale.

NB metodologico: nella dashboard la stagionalita' va misurata sul periodo IN-SAMPLE per
la "scoperta", e poi verificata OUT-OF-SAMPLE. Le funzioni qui calcolano le statistiche
su qualunque serie di rendimenti passata; e' il chiamante a decidere la finestra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEEKDAY_LABELS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
MONTH_LABELS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]


def _bucket_stats(returns: pd.Series, bucket_key, labels) -> pd.DataFrame:
    """
    Statistiche di un raggruppamento calendariale con t-test per gruppo.

    Args:
        returns:    Serie di rendimenti (index datetime)
        bucket_key: array/Series con l'etichetta di gruppo per ogni osservazione
        labels:     mappa indice->nome leggibile

    Returns:
        DataFrame indicizzato per gruppo con: mean_pct, std_pct, t_stat, p_value,
        n, significant (bool a p<0.05).
    """
    r = returns.dropna()
    df = pd.DataFrame({"ret": r.values, "bucket": np.asarray(bucket_key)})
    rows = []
    for b, grp in df.groupby("bucket"):
        vals = grp["ret"].values
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else np.nan
        # t-test a un campione contro media = 0
        if n > 2 and std > 1e-12:
            t_stat, p_val = stats.ttest_1samp(vals, 0.0)
        else:
            t_stat, p_val = np.nan, np.nan
        label = labels[int(b)] if isinstance(labels, list) else labels.get(b, str(b))
        rows.append({
            "bucket": label,
            "mean_pct": mean * 100,
            "std_pct": std * 100,
            "t_stat": t_stat,
            "p_value": p_val,
            "n": n,
            "significant": bool(p_val < 0.05) if p_val == p_val else False,
        })
    out = pd.DataFrame(rows).set_index("bucket")
    return out


def day_of_week_stats(returns: pd.Series) -> pd.DataFrame:
    """Rendimento medio e significativita' per giorno della settimana."""
    idx = returns.dropna().index
    return _bucket_stats(returns, idx.dayofweek, WEEKDAY_LABELS)


def month_of_year_stats(returns: pd.Series) -> pd.DataFrame:
    """Rendimento medio e significativita' per mese dell'anno."""
    idx = returns.dropna().index
    return _bucket_stats(returns, idx.month - 1, MONTH_LABELS)


def turn_of_month_stats(returns: pd.Series, window: int = 3) -> pd.DataFrame:
    """
    Effetto 'turn of the month': confronta i rendimenti nei giorni a cavallo del
    cambio mese (ultimi `window` e primi `window` giorni di trading) vs il resto.

    Args:
        returns: Serie di rendimenti
        window:  numero di giorni di trading considerati a inizio/fine mese

    Returns:
        DataFrame con due gruppi: 'Turn-of-month' e 'Resto del mese'.
    """
    r = returns.dropna().copy()
    df = pd.DataFrame({"ret": r.values}, index=r.index)
    df["ym"] = df.index.to_period("M")
    is_tom = np.zeros(len(df), dtype=bool)
    pos = 0
    for _, grp in df.groupby("ym"):
        m = len(grp)
        mask = np.zeros(m, dtype=bool)
        mask[:window] = True          # primi giorni del mese
        mask[-window:] = True         # ultimi giorni del mese
        is_tom[pos:pos + m] = mask
        pos += m
    bucket = np.where(is_tom, 0, 1)
    return _bucket_stats(r, bucket, {0: "Turn-of-month", 1: "Resto del mese"})


def significant_month_buckets(returns: pd.Series, alpha: float = 0.10) -> list[int]:
    """
    Restituisce i mesi (0=Gen .. 11=Dic) con bias RIALZISTA statisticamente
    significativo a livello `alpha`. Usato dalla confluenza stagionale nei segnali.
    """
    stats_df = month_of_year_stats(returns)
    favorable = []
    for i, label in enumerate(MONTH_LABELS):
        if label in stats_df.index:
            row = stats_df.loc[label]
            if row["p_value"] == row["p_value"] and row["p_value"] < alpha and row["mean_pct"] > 0:
                favorable.append(i)
    return favorable
