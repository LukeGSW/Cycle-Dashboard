"""
data_fetcher.py — Fetch e caching dei dati EODHD.

La chiave API arriva da st.secrets (mai hardcodata). Il caching Streamlit evita di
consumare chiamate API a ogni interazione con la dashboard.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv_cached(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Scarica e cacha lo storico EOD da EODHD.

    Args:
        ticker:  simbolo EODHD (es. 'SPY.US', 'ENI.MI', 'GSPC.INDX', 'EURUSD.FOREX')
        start:   data inizio 'YYYY-MM-DD'
        end:     data fine   'YYYY-MM-DD'
        api_key: chiave EODHD

    Returns:
        DataFrame con DatetimeIndex e colonne open/high/low/close/volume/adjusted_close.
        DataFrame vuoto se la risposta non contiene dati.
    """
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?from={start}&to={end}&period=d"
        f"&api_token={api_key}&fmt=json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    cols = ["open", "high", "low", "close", "volume", "adjusted_close"]
    df = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def get_price_series(df: pd.DataFrame, column: str = "adjusted_close") -> pd.Series:
    """
    Estrae la serie di prezzo di riferimento (default: adjusted_close, che include
    dividendi e split — corretto per l'analisi ciclica di lungo periodo).
    """
    col = column if column in df.columns else "close"
    return df[col].dropna().astype(float)


@st.cache_data(ttl=86400, show_spinner=False)
def search_symbol_cached(query: str, api_key: str) -> pd.DataFrame:
    """
    Ricerca simboli EODHD (utile per trovare il ticker corretto).

    Returns:
        DataFrame con colonne Code, Name, Exchange, Country, Type (se disponibili).
    """
    url = f"https://eodhd.com/api/search/{query}"
    resp = requests.get(url, params={"api_token": api_key, "fmt": "json"}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data) if data else pd.DataFrame()
