"""
app.py — Dashboard di Analisi Ciclica di Strumenti Finanziari.

Spina dorsale oggettiva (DSP/spettro alla Ehlers + esponente di Hurst + stagionalita'),
segnali ciclici causali, e validazione anti-overfitting (in-sample 2/3, out-of-sample 1/3,
ipotesi nulla random-entry, walk-forward, semaforo). Ogni sezione include un riquadro
"Come si legge" per l'interpretazione corretta.

Deploy: GitHub -> Streamlit Cloud. La chiave EODHD va in Settings -> Secrets.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.data_fetcher import fetch_ohlcv_cached, get_price_series
from src.dsp import high_pass
from src.cycles import (spectral_significance, find_peaks, dominant_period,
                        cycle_oscillator, bandpass_bank)
from src.hurst import rolling_hurst
from src.seasonality import (day_of_week_stats, month_of_year_stats,
                             significant_month_buckets)
from src.signals import (cycle_position, regime_mask_from_hurst, seasonal_mask,
                         apply_confluence)
from src.backtest import (pct_returns, run_backtest, compute_metrics,
                          buy_and_hold_returns, count_trades, equity_curve)
from src.validation import (simple_is_oos_split, locked_holdout_split,
                            evaluate_simple, walk_forward_evaluate,
                            random_entry_null, traffic_light)
from src import charts
from src.registry import (load_registry_file, make_record, registry_to_json,
                         config_hash, PARAM_KEYS)

# ===================================================
# CONFIGURAZIONE PAGINA
# ===================================================
st.set_page_config(
    page_title="Analisi Ciclica | Kriterion Quant",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def how_to_read(text: str):
    """Riquadro interpretativo standard sotto ogni grafico."""
    st.info("📖 **Come si legge** — " + text)


# ===================================================
# API KEY
# ===================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    st.error("🔑 Chiave `EODHD_API_KEY` non trovata. Impostala in "
             "`.streamlit/secrets.toml` (locale) o in Settings → Secrets (Streamlit Cloud).")
    st.stop()


# ===================================================
# FUNZIONI PESANTI CON CACHING
# ===================================================
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_spectrum(price, split_pos, min_p, max_p, hp_period, n_surr, seed):
    """Spettro + significativita' + periodo dominante, calcolati SOLO sull'in-sample."""
    is_log = np.log(np.clip(price.iloc[:split_pos].values, 1e-12, None))
    detr = high_pass(is_log, hp_period)
    periods, power, thr = spectral_significance(detr, min_p, max_p, 300, n_surr, 95, seed)
    peaks = find_peaks(periods, power, thr, top_k=6)
    dom = dominant_period(detr, min_p, max_p)
    return periods, power, thr, peaks, dom


@st.cache_data(ttl=3600, show_spinner=False)
def compute_hurst_cached(price, window, method, step):
    return rolling_hurst(price, window, method, step)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_scalogram(price, min_p, max_p, n_periods, bandwidth):
    periods = np.logspace(np.log10(min_p), np.log10(max_p), n_periods)
    mat = bandpass_bank(price, periods, bandwidth)
    return periods, mat


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_cached(price, position, is_frac, cost_bps, min_trades, alpha,
                    min_sharpe_frac, n_sims, seed):
    return evaluate_simple(price, position, is_frac, cost_bps, min_trades,
                           alpha, min_sharpe_frac, n_sims, seed)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_holdout_cached(price_full, dev_end_pos, dom_period, bandwidth, mode,
                            use_regime, use_season, favorable_months, hurst_window,
                            max_hurst, hurst_step, cost_bps, n_sims, min_trades,
                            alpha, min_sharpe_frac, seed):
    """
    ESAME FINALE: valuta la strategia CONGELATA (periodo e mesi favorevoli fissati sullo
    sviluppo) SOLO sul segmento holdout, mai visto ne' per calibrazione ne' per scelta
    parametri. Filtri causali -> nessun look-ahead. E' l'unico test davvero indipendente.
    """
    # Segnale con parametri congelati, esteso a tutta la serie (per continuita' al confine)
    base_full = cycle_position(price_full, dom_period, bandwidth, mode)
    masks_full = []
    if use_regime:
        h_full = rolling_hurst(price_full, hurst_window, "dfa", hurst_step)
        masks_full.append(regime_mask_from_hurst(h_full, max_hurst))
    if use_season:
        masks_full.append(seasonal_mask(price_full, favorable_months))
    position_full = apply_confluence(base_full, masks_full) if masks_full else base_full

    # Backtest sull'intera serie (lo shift +1 al confine e' corretto) e taglio del solo holdout
    bt_full = run_backtest(price_full, position_full, cost_bps)
    ho_ret = bt_full["strat_returns"].iloc[dev_end_pos:]
    ho_pos = position_full.iloc[dev_end_pos:]
    price_ho = price_full.iloc[dev_end_pos:]

    ho_metrics = compute_metrics(ho_ret)
    bh_ho = buy_and_hold_returns(price_full).iloc[dev_end_pos:]
    bh_ho_metrics = compute_metrics(bh_ho)
    ho_trades = count_trades(ho_pos)

    null = random_entry_null(price_ho, ho_pos, n_sims=n_sims, cost_bps=cost_bps, seed=seed)
    verdict, reasons, _ = traffic_light(ho_metrics, bh_ho_metrics, null, ho_trades,
                                        min_trades=min_trades, alpha=alpha,
                                        min_sharpe_frac=min_sharpe_frac)
    return {
        "ho_metrics": ho_metrics, "bh_ho_metrics": bh_ho_metrics, "ho_trades": ho_trades,
        "null": null, "verdict": verdict, "reasons": reasons,
        "ho_equity": equity_curve(ho_ret), "bh_ho_equity": equity_curve(bh_ho),
        "start": price_ho.index[0], "end": price_ho.index[-1], "n_bars": len(price_ho),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def operational_signal_cached(price_full, dom_period, bandwidth, mode, use_regime,
                              use_season, favorable_months, hurst_window, max_hurst,
                              hurst_step):
    """
    SEGNALE OPERATIVO ATTUALE: valuta la strategia (frozen quando la config e' bloccata)
    su TUTTI i dati fino all'ultima barra reale, e restituisce la posizione/fase corrente.
    Causale -> la posizione all'ultima barra e' decisa a quella chiusura, da tenere dalla
    seduta successiva (convenzione t+1). Diverso dalle sezioni di validazione, che si
    fermano al set di sviluppo.
    """
    base = cycle_position(price_full, dom_period, bandwidth, mode)
    osc = cycle_oscillator(price_full, dom_period, bandwidth)
    regime_on, h_last = True, float("nan")
    if use_regime:
        h = rolling_hurst(price_full, hurst_window, "dfa", hurst_step)
        rmask = regime_mask_from_hurst(h, max_hurst)
        base = base * rmask
        regime_on = bool(rmask.iloc[-1] > 0)
        h_last = float(h.dropna().iloc[-1]) if h.notna().any() else float("nan")
    seasonal_on = True
    if use_season:
        smask = seasonal_mask(price_full, favorable_months)
        base = base * smask
        seasonal_on = bool(smask.iloc[-1] > 0)
    return {
        "position": float(base.iloc[-1]),
        "osc_last": float(osc.iloc[-1]),
        "osc_prev": float(osc.iloc[-2]) if len(osc) > 1 else float(osc.iloc[-1]),
        "regime_on": regime_on, "h_last": h_last, "seasonal_on": seasonal_on,
        "date": price_full.index[-1],
    }


# ===================================================
# SIDEBAR — PARAMETRI
# ===================================================
with st.sidebar:
    st.title("🌀 Parametri")
    st.caption("Analisi ciclica · Kriterion Quant")

    with st.form("params"):
        st.subheader("Strumento")
        ticker = st.text_input("Ticker EODHD", value="SPY.US",
                               help="Es: SPY.US, ENI.MI, GSPC.INDX, EURUSD.FOREX, CL.COMM")
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Da", value=pd.Timestamp("2005-01-01"))
        end_date = c2.date_input("A", value=pd.Timestamp.today())

        st.divider()
        st.subheader("Ricerca del ciclo")
        min_p, max_p = st.slider("Range periodi da cercare (barre)", 5, 250, (15, 120),
                                 help="Sotto = rumore; sopra = trend. Il ciclo dominante "
                                      "viene cercato dentro questo intervallo.")
        hp_period = st.slider("Cutoff detrend (high-pass)", 60, 400, 200,
                              help="Cicli piu' lunghi di questo sono trattati come trend e rimossi.")
        bandwidth = st.slider("Larghezza di banda del filtro", 0.10, 0.60, 0.30, 0.05)

        st.divider()
        st.subheader("Segnale e confluenza")
        mode = st.radio("Modalita' posizione", ["long_flat", "long_short"], horizontal=True,
                        help="long_flat: lungo nella salita del ciclo, flat nella discesa. "
                             "long_short: anche corto nella discesa.")
        use_regime = st.checkbox("Confluenza regime (Hurst basso)", value=True,
                                 help="Opera solo in regime mean-reverting, dove il cycle-timing "
                                      "e' piu' affidabile.")
        hurst_window = st.slider("Finestra Hurst", 100, 400, 200, 10)
        max_hurst = st.slider("Hurst massimo per operare", 0.40, 0.70, 0.55, 0.01)
        use_season = st.checkbox("Confluenza stagionale (mesi favorevoli in-sample)", value=False)

        st.divider()
        st.subheader("Validazione")
        cost_bps = st.slider("Costi per trade (bps)", 0.0, 20.0, 3.0, 0.5,
                             help="Commissioni + slippage per unita' di turnover. Il null e il "
                                  "backtest sono SEMPRE al netto di questi costi.")
        alpha = st.slider("Soglia p-value vs null", 0.05, 0.50, 0.25, 0.05,
                          help="Clemente per scelta: 0.25 = la strategia deve stare sopra il "
                               "75° percentile del null random-entry.")
        min_trades = st.slider("Trade OOS minimi per il verde", 5, 60, 20)
        min_sharpe_frac = st.slider("Sharpe OOS minimo (frazione del buy&hold)", 0.0, 1.5, 0.5, 0.1)
        n_sims = st.select_slider("Simulazioni nulle", [100, 200, 300, 500], value=200)
        n_folds = st.slider("Fold walk-forward", 2, 8, 3)
        holdout_frac = st.slider("Holdout finale bloccato (frazione)", 0.0, 0.30, 0.15, 0.05)

        st.divider()
        st.subheader("Output opzionali (CPU)")
        show_scalogram = st.checkbox("Mostra scalogramma", value=False,
                                     help="Heatmap tempo-periodo: informativa ma piu' pesante.")
        run_wf = st.checkbox("Esegui walk-forward", value=True,
                             help="Disattivalo per un'analisi piu' leggera.")

        st.divider()
        st.subheader("🔒 Esame finale")
        unlock_holdout = st.checkbox("🔓 Sblocca l'esame sull'holdout", value=False,
                                     help="Valuta la strategia CONGELATA solo sul segmento "
                                          "holdout mai visto. Da usare UNA VOLTA SOLA, a "
                                          "sviluppo concluso: dopo, il segmento e' 'speso'.")

        st.divider()
        submitted = st.form_submit_button("▶️ Esegui / Aggiorna analisi",
                                          width='stretch', type="primary")

    st.caption("📡 Dati: EODHD | Analisi causale, niente look-ahead")

# ===================================================
# GATE — calcola solo su richiesta (evita il throttle CPU di Streamlit Cloud)
# ===================================================
if submitted:
    st.session_state["ran"] = True
if not st.session_state.get("ran"):
    st.title("🌀 Dashboard di Analisi Ciclica")
    st.info("👈 Imposta i parametri nella sidebar e premi **▶️ Esegui / Aggiorna analisi**.\n\n"
            "Il calcolo (spettro, Hurst, ipotesi nulla, walk-forward) parte **solo quando "
            "premi il pulsante**: cosi' la dashboard non ricalcola a ogni spostamento di slider "
            "e non satura la CPU di Streamlit Community Cloud.")
    st.stop()


# ===================================================
# REGISTRO — richiamo automatico dei parametri validati per ticker
# ===================================================
registry = st.session_state.setdefault("registry", load_registry_file())
ticker_key = ticker.strip().upper()
preset = registry.get(ticker_key)
edit_mode = st.session_state.get(f"edit::{ticker_key}", False)
locked = (preset is not None) and (not edit_mode)

if locked:
    # I parametri BLOCCATI del registro vincono sui widget (ignorati finche' non fai Reset).
    p = preset.get("params", {})
    if p.get("start_date"):
        start_date = pd.to_datetime(p["start_date"]).date()
    # NB: end_date NON e' congelata -> puoi estenderla in avanti per monitorare la strategia
    # congelata su barre nuove senza dover fare Reset.
    min_p = p.get("min_p", min_p)
    max_p = p.get("max_p", max_p)
    hp_period = p.get("hp_period", hp_period)
    bandwidth = p.get("bandwidth", bandwidth)
    mode = p.get("mode", mode)
    use_regime = p.get("use_regime", use_regime)
    hurst_window = p.get("hurst_window", hurst_window)
    max_hurst = p.get("max_hurst", max_hurst)
    use_season = p.get("use_season", use_season)
    cost_bps = p.get("cost_bps", cost_bps)
    holdout_frac = p.get("holdout_frac", holdout_frac)
    min_trades = p.get("min_trades", min_trades)
    alpha = p.get("alpha", alpha)
    min_sharpe_frac = p.get("min_sharpe_frac", min_sharpe_frac)
    n_sims = p.get("n_sims", n_sims)
    n_folds = p.get("n_folds", n_folds)

if preset is not None:
    v = preset.get("holdout_result", {}).get("verdict", "?")
    if locked:
        cR1, cR2 = st.columns([3, 1])
        cR1.success(f"🔒 **{ticker_key} — config VALIDATA e bloccata** "
                    f"(il {str(preset.get('locked_at', '?'))[:10]}, holdout: {v}). "
                    "I parametri sono caricati dal registro: la sidebar e' **ignorata** finche' "
                    "non premi Reset. Puoi cambiare la **data di fine** (o il ticker) per "
                    "**monitorare** la config congelata su barre nuove.")
        if cR2.button("🔓 Reset / ri-valida", width='stretch',
                      help="Sblocca i parametri per una NUOVA validazione. Il record attuale "
                           "sara' archiviato in cronologia al prossimo salvataggio."):
            st.session_state[f"edit::{ticker_key}"] = True
            st.rerun()
    else:
        st.warning(f"✏️ **{ticker_key} in ri-validazione.** Stai usando i parametri della "
                   "sidebar; la vecchia config validata sara' **archiviata** e sostituita solo "
                   "quando sbloccherai un nuovo esame holdout. Per annullare, ricarica la pagina.")

st.download_button("💾 Scarica presets.json (registro attuale) — poi committalo su GitHub",
                   data=registry_to_json(registry), file_name="presets.json",
                   mime="application/json", width='stretch', key="dl_top")


# ===================================================
# HEADER
# ===================================================
st.title("🌀 Dashboard di Analisi Ciclica")
st.markdown("""
Individua i cicli dominanti di uno strumento con metodi **oggettivi** (spettro + filtri di
Ehlers), ne deriva **segnali causali**, e li sottopone a una **validazione anti-overfitting**:
calibrazione in-sample (2/3), test out-of-sample (1/3) contro un'**ipotesi nulla** e il
buy & hold, con **semaforo** di accettazione.

> **Come si usa:** imposta strumento e parametri nella sidebar. Scorri le sezioni dall'alto
> in basso: spettro → ciclo → regime → stagionalita' → **validazione** → verdetto.
""")

with st.expander("📚 Impianto teorico e metodologico (leggere prima)"):
    st.markdown("""
**Spina dorsale oggettiva.** Non ci basiamo su Gann o Elliott (troppo soggettivi e
difficili da falsificare) come *motore* dei segnali. Usiamo:
- **Analisi spettrale** (Lomb-Scargle, robusto ai gap del weekend) per *misurare* i cicli;
- **Filtri causali di John Ehlers** (*Cycle Analytics for Traders*) — passa-banda, roofing,
  sinewave — che sono la traduzione rigorosa e codificabile della teoria dei cicli;
- **Esponente di Hurst** (H. E. Hurst, R/S e DFA) come misuratore di *regime*;
- **Stagionalita'** calendariale, il sottoinsieme *testabile* del folklore ciclico.

**J. M. Hurst e l'analisi ciclica "all'italiana" (Migliorino)** entrano come *overlay
interpretativi*: la loro gerarchia di cicli va **validata contro lo spettro misurato**,
non assunta a priori.

**Anti-overfitting e niente look-ahead (per costruzione):**
1. Il periodo dominante si stima **solo sull'in-sample** e si "congela".
2. Tutti gli indicatori sono **causali** (filtri ricorsivi, non la FFT-Hilbert di scipy che
   sbircia nel futuro). L'esecuzione avviene a **t+1** rispetto alla decisione.
3. L'out-of-sample e' confrontato con un **null random-entry** (stesso numero di trade e
   holding, ingressi casuali) e con il **buy & hold** risk-adjusted: battere l'equity non
   basta, bisogna battere la *fortuna* e il *beta*.
4. **Walk-forward** ancorato (ri-stima del periodo su ogni fold) e **holdout bloccato**.

**Semaforo:** 🟢 supera tutti i criteri · 🟡 positivo ma non batte il null / pochi trade ·
🔴 equity negativa o peggio del null.
""")

st.divider()

# ===================================================
# FETCH DATI
# ===================================================
with st.spinner("⏳ Scaricamento dati da EODHD..."):
    try:
        df = fetch_ohlcv_cached(ticker, str(start_date), str(end_date), EODHD_API_KEY)
    except Exception as e:
        st.error(f"❌ Errore API EODHD: {e}. Verifica ticker, date e chiave.")
        st.stop()

if df.empty or len(df) < 400:
    st.warning(f"⚠️ Dati insufficienti per {ticker} ({len(df)} barre). Servono almeno ~400 "
               "osservazioni per una stima ciclica affidabile. Allarga il periodo o cambia ticker.")
    st.stop()

price_full = get_price_series(df)
n_full = len(price_full)

# HOLDOUT REALE: la coda "congelata" viene TOLTA da tutta l'analisi (spettro, segnale,
# validazione, walk-forward). 'price' = set di SVILUPPO (IS + OOS); tutto il resto lavora
# su 'price'. Con Holdout = 0 il set di sviluppo coincide con l'intera serie e l'OOS si
# allarga (piu' barre -> piu' trade).
if holdout_frac > 0:
    dev_end_pos, holdout_start_date = locked_holdout_split(price_full, holdout_frac)
else:
    dev_end_pos, holdout_start_date = n_full, None

price = price_full.iloc[:dev_end_pos]           # set di sviluppo: qui gira tutta l'analisi
holdout_price = price_full.iloc[dev_end_pos:]   # coda congelata, mai usata per lo scoring

split_pos, split_date = simple_is_oos_split(price, is_frac=2 / 3)
# Nel grafico del set di sviluppo la zona holdout e' vuota (l'holdout e' oltre 'price').
chart_holdout_date = price.index[-1]

# KPI (descrivono lo strumento e il set di sviluppo effettivamente analizzato)
returns_all = pct_returns(price)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prezzo (ultimo reale)", f"{price_full.iloc[-1]:,.2f}")
c2.metric("Barre sviluppo", f"{len(price):,}")
c3.metric("Periodo sviluppo", f"{price.index[0].date()} → {price.index[-1].date()}")
c4.metric("Vol. annua", f"{returns_all.std() * np.sqrt(252) * 100:.1f}%")
c5.metric("Buy&hold (sviluppo)", f"{(price.iloc[-1] / price.iloc[0] - 1) * 100:+.0f}%")
if holdout_start_date is not None:
    st.caption(f"🔒 {len(holdout_price)} barre in **holdout congelato** dal "
               f"{holdout_start_date.date()} — escluse da spettro, segnale, validazione e "
               f"walk-forward. Imposta **Holdout = 0** per includerle e allargare l'OOS.")
st.divider()

# ===================================================
# SEZIONE 1 — SPETTRO
# ===================================================
st.header("1 · Spettro — quali cicli esistono davvero")
st.markdown("Periodogramma di **Lomb-Scargle** sulla serie detrendizzata, calcolato **solo "
            "sull'in-sample**. La soglia rossa nasce da **surrogati a fase randomizzata**: "
            "un picco che la supera e' improbabile per puro caso.")

with st.spinner("🔬 Analisi spettrale e test di significativita'..."):
    periods, power, thr, peaks, dom_period = analyze_spectrum(
        price, split_pos, min_p, max_p, hp_period, n_surr=int(min(n_sims, 150)), seed=42)

if not (dom_period == dom_period):
    st.error("Impossibile stimare un periodo dominante nel range scelto. Allarga il range.")
    st.stop()

# Se la config e' bloccata, il periodo dominante e' CONGELATO dal registro (strategia
# faithful): lo spettro qui sotto e' comunque ricalcolato sui dati attuali, cosi' si vede
# se il ciclo sta derivando rispetto al periodo validato.
frozen_period = locked and preset.get("frozen", {}).get("dom_period")
if frozen_period:
    dom_period = float(frozen_period)

st.plotly_chart(charts.build_spectrum_chart(periods, power, thr, peaks, dom_period),
                width='stretch')
if frozen_period:
    st.caption(f"🔒 Segnale sul periodo **congelato ~{dom_period:.0f} barre** (dal registro). "
               "Se il picco dello spettro sopra si e' spostato, il ciclo sta derivando dal "
               "valore validato: valutane un Reset e ri-validazione.")
how_to_read(
    "l'asse X e' il **periodo del ciclo in barre** (giorni), l'asse Y la sua **potenza**. "
    "I picchi sopra la **linea rossa tratteggiata** sono cicli statisticamente significativi "
    "(verdi = significativi, grigi = no). La banda arancione segna il **ciclo dominante** "
    f"(~**{dom_period:.0f} barre**), che verra' usato per generare il segnale. "
    "Se *nessun* picco supera la soglia, lo strumento non ha cicli affidabili: aspettati "
    "un semaforo prudente più avanti.")

peaks_df = pd.DataFrame(peaks)
if not peaks_df.empty:
    peaks_df = peaks_df.rename(columns={"period": "Periodo (barre)", "power": "Potenza",
                                        "significant": "Significativo"})
    peaks_df["Periodo (barre)"] = peaks_df["Periodo (barre)"].round(0)
    peaks_df["Potenza"] = peaks_df["Potenza"].round(3)
    st.dataframe(peaks_df, width='stretch', hide_index=True)

st.divider()

# ===================================================
# SEZIONE 2 — CICLO ISOLATO + SCALOGRAMMA
# ===================================================
st.header("2 · Il ciclo isolato e la sua deriva nel tempo")
osc = cycle_oscillator(price, dom_period, bandwidth)

st.plotly_chart(
    charts.build_price_cycle_chart(price, osc, split_date, chart_holdout_date, ticker),
    width='stretch')
how_to_read(
    "in alto il **prezzo** con le tre zone: **In-Sample** (blu, dove si calibra), "
    "**Out-of-Sample** (arancio, dove si testa in cieco), **Holdout** (viola, da guardare "
    "una volta sola). In basso l'**oscillatore ciclico** normalizzato: vicino a **-1** = "
    "minimo del ciclo (potenziale acquisto), vicino a **+1** = massimo (potenziale uscita). "
    "È l'onda che il segnale cerca di cavalcare.")

if show_scalogram:
    with st.spinner("🌊 Calcolo scalogramma..."):
        sc_periods, sc_mat = compute_scalogram(price, min_p, max_p, 30, bandwidth)
    st.plotly_chart(charts.build_scalogram(sc_mat, sc_periods, price.index),
                    width='stretch')
    how_to_read(
        "questa heatmap mostra, per ogni **data** (X) e ogni **periodo** (Y), quanta **energia "
        "ciclica** c'e' (colore chiaro = molta). Se la banda luminosa **si sposta** in verticale "
        "nel tempo, il ciclo dominante **cambia periodo** (non stazionarieta'): e' il motivo per "
        "cui ri-stimiamo il periodo nel walk-forward invece di fidarci di un valore fisso.")
else:
    st.caption("🌊 Scalogramma disattivato — attivalo nella sidebar (Output opzionali) se ti "
               "serve vedere la deriva del periodo nel tempo.")

st.divider()

# ===================================================
# SEZIONE 3 — FASE ATTUALE
# ===================================================
st.header("3 · Fase del ciclo (a fine set di sviluppo)")
osc_last = float(osc.iloc[-1])
rising = bool(osc.iloc[-1] > osc.iloc[-2])
if osc_last < -0.4:
    base_state = "vicino al MINIMO"
elif osc_last > 0.4:
    base_state = "vicino al MASSIMO"
else:
    base_state = "in zona intermedia"
state_label = f"{base_state} · {'in salita' if rising else 'in discesa'}"

colA, colB = st.columns([1, 1.3])
with colA:
    st.plotly_chart(charts.build_phase_gauge(osc_last, state_label), width='stretch')
with colB:
    st.markdown(f"""
    **Fase alla fine del set di sviluppo ({price.index[-1].date()}):**

    - Posizione nel ciclo: **{osc_last:+.2f}** → *{base_state}*
    - Direzione: **{'in salita ⬆️' if rising else 'in discesa ⬇️'}**

    ⚠️ Con **Holdout > 0** questa **non** e' la fase di oggi: si ferma a fine sviluppo. Per il
    **segnale operativo attuale** (ultima barra reale, tutti i dati fino a oggi) vedi la sezione
    **🎯 Segnale operativo attuale** piu' in basso.
    """)
how_to_read(
    "il **gauge** riassume il valore dell'oscillatore adesso. Zona verde (sinistra) = minimo "
    "del ciclo; zona rossa (destra) = massimo; centro = transito. Non e' un segnale di per "
    "se': va sempre incrociato con regime e validazione OOS.")

st.divider()

# ===================================================
# SEZIONE 4 — REGIME (HURST)
# ===================================================
st.header("4 · Regime — il cycle-timing conviene adesso?")
with st.spinner("📐 Esponente di Hurst rolling..."):
    # step adattivo: su serie lunghe calcola 1 barra ogni N (l'Hurst varia lentamente)
    hurst_step = max(1, len(price) // 1500)
    hurst = compute_hurst_cached(price, hurst_window, "dfa", hurst_step)
h_last = hurst.dropna().iloc[-1] if hurst.notna().any() else np.nan

st.plotly_chart(charts.build_hurst_chart(price, hurst), width='stretch')
if h_last == h_last:
    regime_txt = ("**mean-reverting** (favorevole ai cicli)" if h_last < 0.5
                  else "**trending** (cicli meno affidabili)")
    st.markdown(f"Esponente di Hurst attuale: **{h_last:.2f}** → regime {regime_txt}.")
how_to_read(
    "l'esponente di Hurst misura la **memoria** della serie. **H < 0.5** (fascia verde) = "
    "serie che tende a **tornare sulla media**: terreno ideale per comprare i minimi e "
    "vendere i massimi del ciclo. **H > 0.5** (fascia rossa) = serie **persistente/in trend**, "
    "dove il timing dei cicli sbaglia più spesso. La confluenza di regime (sidebar) spegne il "
    "segnale quando H sale sopra la tua soglia.")

st.divider()

# ===================================================
# SEZIONE 5 — STAGIONALITA'
# ===================================================
st.header("5 · Stagionalita' — il ciclo del calendario")
is_returns = returns_all.iloc[:split_pos]
dow = day_of_week_stats(is_returns)
moy = month_of_year_stats(is_returns)
favorable_months = significant_month_buckets(is_returns, alpha=0.10)
# Config bloccata: mesi favorevoli CONGELATI dal registro (strategia faithful)
if locked and "favorable_months" in preset.get("frozen", {}):
    favorable_months = [int(m) for m in preset["frozen"]["favorable_months"]]

cS1, cS2 = st.columns(2)
with cS1:
    st.plotly_chart(charts.build_seasonality_bars(dow, "Rendimento medio per giorno"),
                    width='stretch')
with cS2:
    st.plotly_chart(charts.build_seasonality_bars(moy, "Rendimento medio per mese"),
                    width='stretch')
how_to_read(
    "ogni barra e' il **rendimento medio** del bucket (giorno o mese) **in-sample**. Le barre "
    "con la **stella ★** (verdi/rosse) sono **statisticamente significative** (t-test, p<0.05); "
    "le grigie sono rumore da ignorare. Solo gli effetti con la stella meritano attenzione: "
    "e' cosi' che si evita di scambiare la casualita' per stagionalita'.")
if favorable_months:
    from src.seasonality import MONTH_LABELS
    st.caption("Mesi con bias rialzista significativo (usati dalla confluenza stagionale se "
               "attiva): " + ", ".join(MONTH_LABELS[m] for m in favorable_months))

st.divider()

# ===================================================
# SEZIONE 6 — SEGNALE + CONFLUENZA
# ===================================================
st.header("6 · Costruzione del segnale (con confluenza)")
base_pos = cycle_position(price, dom_period, bandwidth, mode)

masks = []
active_filters = ["Ciclo (sinewave di Ehlers)"]
if use_regime:
    rmask = regime_mask_from_hurst(hurst, max_hurst)  # riusa l'Hurst gia' calcolato
    masks.append(rmask)
    active_filters.append(f"Regime (Hurst ≤ {max_hurst})")
if use_season:
    smask = seasonal_mask(price, favorable_months)
    masks.append(smask)
    active_filters.append("Stagionalita' (mesi favorevoli)")

position = apply_confluence(base_pos, masks) if masks else base_pos
exposure_pct = (position != 0).mean() * 100

st.markdown(f"**Filtri in confluenza attivi:** {' · '.join(active_filters)}  \n"
            f"**Tempo in mercato:** {exposure_pct:.0f}% delle barre · "
            f"**Modalita':** `{mode}`")
how_to_read(
    "il segnale nasce dal **giro del ciclo** (crossover *sinewave* di Ehlers) e viene poi "
    "**filtrato per confluenza**: la posizione sopravvive solo dove *tutti* i filtri attivi "
    "concordano. Più filtri indipendenti (spettro + regime + stagionalita') = meno probabilità "
    "di adattarsi al rumore. Un tempo-in-mercato molto basso significa che i filtri sono "
    "restrittivi: attenzione al numero di trade OOS.")

st.divider()

# ===================================================
# SEGNALE OPERATIVO ATTUALE (ultima barra reale, tutti i dati fino a oggi)
# ===================================================
st.header("🎯 Segnale operativo attuale")
with st.spinner("🎯 Calcolo del segnale all'ultima barra reale..."):
    opsig = operational_signal_cached(
        price_full, dom_period, bandwidth, mode, use_regime, use_season,
        favorable_months, hurst_window, max_hurst, max(1, len(price_full) // 1500))

pos_val = opsig["position"]
pos_label = {1.0: "🟢 LONG", 0.0: "⚪ FLAT", -1.0: "🔴 SHORT"}.get(pos_val, f"{pos_val:+.0f}")
osc_v = opsig["osc_last"]
rising = osc_v > opsig["osc_prev"]
near = "vicino al MINIMO" if osc_v < -0.4 else "vicino al MASSIMO" if osc_v > 0.4 else "zona intermedia"

cO1, cO2, cO3 = st.columns(3)
cO1.metric("Posizione (dalla prossima seduta)", pos_label)
cO2.metric("Fase del ciclo", f"{osc_v:+.2f}", delta=("in salita" if rising else "in discesa"),
           delta_color="off")
cO3.metric("Ultima barra reale", f"{opsig['date'].date()}")

filt_lines = []
if use_regime:
    filt_lines.append(
        f"**Regime** (Hurst {opsig['h_last']:.2f} vs soglia {max_hurst}): "
        + ("✅ operativo" if opsig["regime_on"] else "⛔ spento — mercato troppo in trend, "
           "la posizione ciclica e' forzata a FLAT"))
if use_season:
    filt_lines.append("**Stagionalità**: "
                      + ("✅ mese favorevole" if opsig["seasonal_on"] else "⛔ mese non favorevole"))
if filt_lines:
    st.markdown("**Stato dei filtri all'ultima barra:**\n\n"
                + "\n".join("- " + ln for ln in filt_lines))

colg1, colg2 = st.columns([1, 1.2])
with colg1:
    st.plotly_chart(charts.build_phase_gauge(
        osc_v, f"{near} · {'in salita' if rising else 'in discesa'}"), width='stretch')
with colg2:
    if locked:
        st.success(f"🔒 Strategia **CONGELATA dal registro**, valutata su tutti i dati fino al "
                   f"**{opsig['date'].date()}**. Questo e' il segnale da seguire operativamente: "
                   f"**{pos_label}** dalla prossima seduta.")
    else:
        st.warning("⚠️ Parametri di **sviluppo** (non ancora validati/bloccati). Questo segnale "
                   "e' solo indicativo: completa la validazione (esame holdout) e blocca la "
                   "config prima di operare.")

how_to_read(
    "questa e' l'**unica sezione che guarda l'ultima barra reale** (le sezioni di validazione "
    "qui sotto si fermano al set di sviluppo, escludendo l'holdout). La **posizione** e' quella "
    "decisa alla chiusura piu' recente, da assumere dalla **prossima seduta** (esecuzione t+1). "
    "Con Holdout > 0 e config bloccata, la strategia congelata viene comunque calcolata fino a "
    "oggi: e' qui che leggi cosa fare adesso.")

st.divider()

# ===================================================
# SEZIONE 7 — VALIDAZIONE OOS + SEMAFORO
# ===================================================
st.header("7 · Validazione Out-of-Sample")
with st.spinner("🧪 Backtest, ipotesi nulla e semaforo..."):
    ev = evaluate_cached(price, position, 2 / 3, cost_bps, min_trades, alpha,
                         min_sharpe_frac, int(n_sims), 42)

verdict = ev["verdict"]
verdict_map = {
    "GREEN": ("🟢 VERDE", "success", "Segnali futuri accettabili."),
    "YELLOW": ("🟡 GIALLO", "warning", "Solo paper trading / watchlist."),
    "RED": ("🔴 ROSSO", "error", "Segnale da scartare."),
}
label, box, tagline = verdict_map[verdict]
banner = getattr(st, box)
st.markdown("**Esito del singolo split 2/3–1/3.** Il verdetto *finale* (Sezione 9) tiene conto "
            "anche del walk-forward, per non fidarsi di un'unica finestra.")
banner(f"### {label} — {tagline}")
for r in ev["reasons"]:
    st.write("• " + r)

oos_bars = ev["oos_metrics"]["n_periods"]
st.caption(f"🎯 Finestra OOS testata: **{ev['split_date'].date()} → {price.index[-1].date()}** "
           f"({oos_bars} barre, {ev['n_trades_oos']} trade). Cambiando l'Holdout questa finestra "
           "si **sposta nel tempo**: se un piccolo spostamento del confine ribalta il verdetto, "
           "l'edge non e' robusto (ecco perche' il verdetto finale pesa anche il walk-forward).")

# Metriche IS vs OOS vs Buy&Hold
def _fmt(m):
    return {
        "Rendim. totale": f"{m['total_return']*100:+.1f}%",
        "CAGR": f"{m['cagr']*100:+.1f}%" if m['cagr'] == m['cagr'] else "—",
        "Vol. annua": f"{m['ann_vol']*100:.1f}%" if m['ann_vol'] == m['ann_vol'] else "—",
        "Sharpe": f"{m['sharpe']:.2f}",
        "Sortino": f"{m['sortino']:.2f}" if m['sortino'] == m['sortino'] else "—",
        "Max DD": f"{m['max_dd']*100:.1f}%",
        "Hit rate": f"{m['hit_rate']*100:.0f}%" if m['hit_rate'] == m['hit_rate'] else "—",
    }

metrics_table = pd.DataFrame({
    "In-Sample (strategia)": _fmt(ev["is_metrics"]),
    "Out-of-Sample (strategia)": _fmt(ev["oos_metrics"]),
    "Out-of-Sample (Buy & Hold)": _fmt(ev["bh_oos_metrics"]),
}).T
st.dataframe(metrics_table, width='stretch')
how_to_read(
    "confronta le tre righe. La strategia deve **reggere fuori campione**: se l'IS e' ottimo "
    "ma l'OOS crolla, e' overfitting. La riga **Buy & Hold OOS** e' il metro del *beta*: uno "
    "Sharpe OOS della strategia molto sotto il buy&hold significa che stai solo cavalcando il "
    "mercato con più rischio operativo.")

cE1, cE2 = st.columns([1.4, 1])
with cE1:
    st.plotly_chart(charts.build_equity_chart(ev["equity"],
                                              (1 + ev["bh_returns"]).cumprod(), split_date),
                    width='stretch')
    how_to_read(
        "equity a **base 100**. La linea verticale segna l'inizio dell'**OOS**: e' *lì* che "
        "conta la performance. A sinistra la strategia e' calibrata (facile sembrare brava); "
        "a destra e' il vero esame.")
with cE2:
    st.plotly_chart(charts.build_drawdown_chart(ev["equity"]), width='stretch')
    how_to_read("il **drawdown** e' il calo dal massimo precedente. Misura quanto dolore "
                "avresti sopportato: un rendimento buono con drawdown enorme e' spesso "
                "intollerabile nella realta'.")

# Distribuzione nulla
null = ev["null"]
if null["ret_dist"].size:
    st.subheader("Test di significativita' contro l'ipotesi nulla")
    st.plotly_chart(
        charts.build_null_distribution(null["ret_dist"], null["strat_total"],
                                       null["p_value_return"], "Rendimento OOS"),
        width='stretch')
    how_to_read(
        "l'istogramma grigio e' cio' che avrebbero reso **migliaia di strategie con ingressi "
        "casuali** ma con lo *stesso numero di trade e la stessa durata* della tua, sullo "
        "stesso prezzo. La linea **verde** e' la tua strategia. Se sta a **destra** della "
        "massa (oltre il 75° percentile arancione), il tuo **timing** aggiunge valore: "
        f"qui **p = {null['p_value_return']:.2f}** "
        f"({'supera' if null['p_value_return'] <= alpha else 'NON supera'} la soglia {alpha}). "
        "Se cade **dentro** la massa, i tuoi guadagni sono spiegabili dal caso + deriva.")

st.divider()

# ===================================================
# SEZIONE 8 — WALK-FORWARD
# ===================================================
st.header("8 · Walk-Forward — stabilita' su piu' finestre")
st.markdown("Invece di un solo taglio 2/3–1/3, il periodo dominante viene **ri-stimato su "
            "ogni fold** (train espandente) e testato sulla finestra successiva. Cosi' si "
            "vede se il ciclo e' *stabile* o solo un colpo di fortuna di un periodo.")


def signal_builder(train_end_pos: int) -> pd.Series:
    """Ricalibra periodo dominante E mesi stagionali SOLO su [0, train_end_pos), poi genera
    la posizione. Il regime (Hurst rolling) e' gia' causale, quindi la sua maschera si puo'
    riusare; i mesi favorevoli, invece, vanno RI-SELEZIONATI per fold: usare quelli globali
    (stimati sul 2/3 in-sample) introdurrebbe look-ahead nei fold OOS piu' precoci."""
    tp = price.iloc[:train_end_pos]
    is_log = np.log(np.clip(tp.values, 1e-12, None))
    detr = high_pass(is_log, hp_period)
    per = dominant_period(detr, min_p, max_p)
    if not (per == per):
        per = dom_period
    base = cycle_position(price, per, bandwidth, mode)

    fold_masks = []
    if use_regime:
        fold_masks.append(rmask)  # causale: riutilizzabile senza look-ahead
    if use_season:
        fold_fav = significant_month_buckets(returns_all.iloc[:train_end_pos], alpha=0.10)
        fold_masks.append(seasonal_mask(price, fold_fav))
    return apply_confluence(base, fold_masks) if fold_masks else base


wf = {"folds": [], "consistency": 0.0}
if run_wf:
    with st.spinner("🔁 Walk-forward in corso..."):
        wf = walk_forward_evaluate(price, signal_builder, n_folds=int(n_folds),
                                   min_train_frac=0.5, cost_bps=cost_bps)

if wf["folds"]:
    st.plotly_chart(charts.build_walkforward_bars(wf["folds"]), width='stretch')
    consistency = wf["consistency"]
    cwf1, cwf2 = st.columns([1, 2])
    cwf1.metric("Fold OOS positivi", f"{consistency*100:.0f}%")
    cwf2.markdown("Consistenza **≥ 50%** dei fold positivi = segnale ragionevolmente stabile. "
                  "Un solo fold molto positivo e gli altri negativi = **fragilita'** "
                  "(evita di fidarti).")
    how_to_read(
        "ogni barra e' il **rendimento OOS di un fold** (una finestra futura mai vista in "
        "calibrazione). Vuoi vedere **più barre verdi** e di dimensioni simili: significa che "
        "il ciclo funziona in epoche diverse. Barre alternate verde/rosso = ciclo instabile.")
elif not run_wf:
    st.caption("🔁 Walk-forward disattivato — attivalo nella sidebar per verificare la "
               "stabilita' del ciclo su piu' finestre OOS.")

st.divider()

# ===================================================
# SEZIONE 9 — VERDETTO E HOLDOUT
# ===================================================
st.header("9 · Verdetto finale")

# Verdetto finale ROBUSTO: un verde del singolo split viene declassato a giallo se il
# walk-forward e' instabile (< 50% di fold positivi). Un solo split fortunato non basta.
final_verdict = verdict
downgrade_reason = None
if verdict == "GREEN" and run_wf and wf["folds"] and wf["consistency"] < 0.5:
    final_verdict = "YELLOW"
    downgrade_reason = (
        f"Il singolo split OOS e' verde, ma il **walk-forward e' instabile** "
        f"(solo {wf['consistency']*100:.0f}% di fold positivi): il verde dipende "
        "probabilmente da **un'unica finestra fortunata**, non da un ciclo stabile. "
        "Declassato a GIALLO.")

f_label, f_box, f_tagline = verdict_map[final_verdict]
getattr(st, f_box)(f"### {f_label} — {f_tagline}")
if downgrade_reason:
    st.write("• " + downgrade_reason)
elif final_verdict == "GREEN" and run_wf and wf["folds"]:
    st.write(f"• Coerente anche col walk-forward ({wf['consistency']*100:.0f}% di fold positivi).")
elif verdict == "GREEN" and (not run_wf or not wf["folds"]):
    st.caption("ℹ️ Walk-forward disattivato: il verde si basa solo sul singolo split. "
               "Attivalo per un verdetto robusto alla scelta della finestra OOS.")

cV1, cV2, cV3 = st.columns(3)
cV1.metric("p-value vs null", f"{null['p_value_return']:.2f}",
           help="≤ soglia = timing significativo")
cV2.metric("Trade OOS", f"{ev['n_trades_oos']}",
           delta=f"min {min_trades}", delta_color="normal")
cV3.metric("Consistenza WF", f"{wf['consistency']*100:.0f}%" if wf["folds"] else "—")

if holdout_start_date is not None:
    st.warning(f"🔒 **Holdout bloccato:** le ultime **{len(holdout_price)} barre** (dal "
               f"**{holdout_start_date.date()}**, {holdout_frac*100:.0f}% finale) sono "
               "**escluse da tutta l'analisi** e vanno guardate **una sola volta**, a fine "
               "sviluppo. Ogni volta che ritocchi i parametri guardando l'OOS, l'OOS diventa "
               "in-sample: l'holdout e' l'ultima difesa contro il data-snooping. "
               "Con **Holdout = 0** rientrano nel set e l'OOS si allarga (piu' trade).")

# ===================================================
# SEZIONE 10 — ESAME FINALE SULL'HOLDOUT
# ===================================================
if holdout_start_date is not None:
    st.divider()
    st.header("10 · 🔒 Esame finale sull'holdout")
    if not unlock_holdout:
        st.info("Questo e' l'**esame indipendente definitivo**: la strategia congelata sullo "
                "sviluppo, valutata sul segmento holdout che non ha mai influenzato nulla "
                "(ne' calibrazione ne' scelta parametri). E' **bloccato**: sbloccalo dalla "
                "sidebar (🔒 Esame finale) **solo a sviluppo concluso** e guardalo **una volta "
                "sola**. Se lo consulti e poi ritocchi i parametri, l'holdout e' bruciato.")
    else:
        st.error("🔓 **Holdout sbloccato — segmento ora 'speso'.** Usa questo esito come "
                 "decisione **finale** (accetta / scarta il segnale), NON per iterare: se "
                 "ritocchi i parametri guardando questo risultato, stai facendo overfitting "
                 "sull'ultima difesa.")
        with st.spinner("🔒 Valutazione della strategia congelata sull'holdout..."):
            hoev = evaluate_holdout_cached(
                price_full, dev_end_pos, dom_period, bandwidth, mode, use_regime, use_season,
                favorable_months, hurst_window, max_hurst, max(1, len(price_full) // 1500),
                cost_bps, int(n_sims), min_trades, alpha, min_sharpe_frac, 42)

        h_label, h_box, h_tag = verdict_map[hoev["verdict"]]
        getattr(st, h_box)(f"### Holdout: {h_label} — {h_tag}")
        for r in hoev["reasons"]:
            st.write("• " + r)
        st.caption(f"Segmento holdout: **{hoev['start'].date()} → {hoev['end'].date()}** "
                   f"({hoev['n_bars']} barre, {hoev['ho_trades']} trade). Strategia congelata: "
                   f"periodo ~{dom_period:.0f} barre e mesi favorevoli fissati sullo sviluppo.")

        ho_tbl = pd.DataFrame({
            "Strategia congelata (holdout)": _fmt(hoev["ho_metrics"]),
            "Buy & Hold (holdout)": _fmt(hoev["bh_ho_metrics"]),
        }).T
        st.dataframe(ho_tbl, width='stretch')

        st.plotly_chart(charts.build_holdout_equity(hoev["ho_equity"], hoev["bh_ho_equity"]),
                        width='stretch')

        ho_null = hoev["null"]
        if ho_null["ret_dist"].size:
            st.plotly_chart(
                charts.build_null_distribution(ho_null["ret_dist"], ho_null["strat_total"],
                                               ho_null["p_value_return"], "Rendimento holdout"),
                width='stretch')

        how_to_read(
            "questo e' l'**unico test su dati che non hanno MAI influenzato la strategia**. "
            "Verde qui — positivo netto costi, batte il null e regge il confronto risk-adjusted "
            "col buy&hold — e' la **conferma piu' forte** che l'edge generalizza. Se non regge, "
            "fuori campione l'edge non ha tenuto: la decisione corretta e' **scartare**, non "
            "tornare a ottimizzare (bruceresti l'ultima difesa).")

        # --- REGISTRAZIONE della config validata (salvataggio allo sblocco dell'esame) ---
        if not locked:
            reg = st.session_state["registry"]
            current_params = {
                "start_date": str(start_date), "end_date": str(end_date),
                "min_p": int(min_p), "max_p": int(max_p), "hp_period": int(hp_period),
                "bandwidth": float(bandwidth), "mode": mode, "use_regime": bool(use_regime),
                "hurst_window": int(hurst_window), "max_hurst": float(max_hurst),
                "use_season": bool(use_season), "cost_bps": float(cost_bps),
                "holdout_frac": float(holdout_frac), "min_trades": int(min_trades),
                "alpha": float(alpha), "min_sharpe_frac": float(min_sharpe_frac),
                "n_sims": int(n_sims), "n_folds": int(n_folds),
            }
            frozen = {"dom_period": float(dom_period),
                      "favorable_months": [int(m) for m in favorable_months]}
            hm = hoev["ho_metrics"]
            holdout_result = {
                "verdict": hoev["verdict"],
                "total_return": float(hm["total_return"]), "sharpe": float(hm["sharpe"]),
                "max_dd": float(hm["max_dd"]),
                "p_value": float(hoev["null"]["p_value_return"]),
                "n_trades": int(hoev["ho_trades"]),
            }
            new_hash = config_hash(current_params)
            old = reg.get(ticker_key)
            if old is None or old.get("config_hash") != new_hash:
                reg[ticker_key] = make_record(
                    current_params, frozen, holdout_result,
                    datetime.now().isoformat(timespec="seconds"), old_record=old)
                st.session_state["registry"] = reg
                st.session_state[f"edit::{ticker_key}"] = False   # ri-blocca dopo il salvataggio
                st.success(f"🔒 Config di **{ticker_key}** registrata e bloccata "
                           f"(holdout: {hoev['verdict']}). Scarica il registro aggiornato qui "
                           "sotto e **committalo su GitHub** per renderlo permanente.")
                st.download_button(
                    "💾 Scarica presets.json aggiornato (committalo su GitHub)",
                    data=registry_to_json(reg), file_name="presets.json",
                    mime="application/json", width='stretch', key="dl_after_save")
            else:
                st.session_state[f"edit::{ticker_key}"] = False  # identica a registro: ri-blocca
                st.caption(f"Config di {ticker_key} gia' a registro (hash {new_hash}).")
        else:
            st.caption("Config gia' registrata e bloccata. Per rivalidare, usa **Reset** in alto.")

st.divider()
st.caption("⚠️ Strumento di ricerca a scopo educativo. Nessun risultato passato garantisce "
           "performance future. Non e' consulenza finanziaria. · Kriterion Quant")
