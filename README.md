# 🌀 Dashboard di Analisi Ciclica

Dashboard Streamlit per l'**analisi ciclica di strumenti finanziari** con dati **EODHD**.
Individua i cicli dominanti con metodi *oggettivi* (analisi spettrale + filtri causali di
John Ehlers), ne deriva segnali *causali* (niente look-ahead) e li sottopone a una
**validazione anti-overfitting** con semaforo di accettazione.

---

## Filosofia

Non usiamo Gann o Elliott come *motore* dei segnali (troppo soggettivi e difficili da
falsificare). La spina dorsale è **oggettiva e replicabile**:

| Livello | Metodo | Ruolo |
|---|---|---|
| Misura dei cicli | Periodogramma di **Lomb-Scargle** + significatività via surrogati | *scoperta* |
| Filtri | Passa-banda / roofing / **sinewave** di **Ehlers** (causali) | *segnale* |
| Regime | Esponente di **Hurst** (R/S, DFA) | *quando* operare |
| Calendario | Stagionalità con t-test | *conferma* |
| J. M. Hurst / Migliorino | gerarchia ciclica | *overlay interpretativo, validato* |

## Anti-overfitting (per costruzione)

1. **Periodo dominante stimato solo in-sample** e "congelato".
2. **Tutti gli indicatori sono causali** (filtri ricorsivi di Ehlers, non la FFT-Hilbert di
   scipy che sbircia nel futuro). Esecuzione a **t+1** rispetto alla decisione.
3. **Ipotesi nulla random-entry**: la strategia deve battere migliaia di strategie con lo
   stesso numero di trade/holding ma ingressi casuali (batti la *fortuna*), **e** il buy &
   hold risk-adjusted (batti il *beta*).
4. **Walk-forward** ancorato (ri-stima del periodo su ogni fold) + **holdout bloccato**.

**Semaforo:** 🟢 supera tutti i criteri OOS · 🟡 positivo ma non batte il null / pochi trade ·
🔴 equity negativa o peggio del null.

---

## Struttura

```
cycle-dashboard/
├── app.py                  # UI Streamlit + riquadri "Come si legge"
├── requirements.txt
├── presets.json            # registro dei parametri validati per ticker (versionato in git)
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── src/
    ├── data_fetcher.py     # fetch/caching EODHD
    ├── dsp.py              # filtri causali di Ehlers
    ├── cycles.py           # spettro + oscillatore/fase causali
    ├── hurst.py            # esponente di Hurst (regime)
    ├── seasonality.py      # stagionalità + significatività
    ├── signals.py          # segnale ciclico + confluenza
    ├── backtest.py         # backtest causale + metriche
    ├── validation.py       # split, walk-forward, null, semaforo
    ├── registry.py         # registro parametri validati (save/recall per ticker)
    └── charts.py           # grafici Plotly dark
```

### Registro dei parametri validati

Quando **sblocchi l'esame finale sull'holdout**, la config (parametri + periodo/mesi
congelati + verdetto holdout + timestamp + hash) viene **registrata e associata al ticker**.
Al successivo caricamento di quel ticker i parametri vengono **richiamati automaticamente** e
**bloccati**: per modificarli serve un **Reset** esplicito (il vecchio record va in `history`).

La persistenza è **git-based** (Streamlit Cloud ha filesystem effimero): l'app legge
`presets.json` dal repo all'avvio; al salvataggio ti offre il file aggiornato da **committare
su GitHub**. La cronologia git è l'audit trail anti-imbroglio.

---

## Avvio in locale

```bash
pip install -r requirements.txt
```

Crea `.streamlit/secrets.toml` (copia da `secrets.toml.example`) con la tua chiave:

```toml
EODHD_API_KEY = "la-tua-chiave-eodhd"
```

Avvia:

```bash
streamlit run app.py
```

---

## Deploy su Streamlit Cloud

1. Push del repository su GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → seleziona il repo e `app.py`.
3. **Advanced settings → Secrets**: incolla `EODHD_API_KEY = "..."`.
4. **Deploy** → app live su `https://<nome>.streamlit.app`.

---

## Formato ticker EODHD

`SPY.US` · `AAPL.US` · `ENI.MI` · `GSPC.INDX` (S&P 500) · `EURUSD.FOREX` · `CL.COMM` (WTI)

---

## Riferimenti

- J. Ehlers, *Cycle Analytics for Traders* (2013); *Rocket Science for Traders* (2001).
- H. E. Hurst, *Long-Term Storage Capacity of Reservoirs* (1951) — esponente di Hurst.
- Peng et al. (1994) — Detrended Fluctuation Analysis (DFA).
- Lomb (1976), Scargle (1982) — periodogramma per dati non uniformi.

---

> ⚠️ Strumento di ricerca a scopo **educativo**. Le performance passate non garantiscono
> risultati futuri. Non è consulenza finanziaria.
