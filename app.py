"""
app.py — Dashboard di Analisi Ciclica di Strumenti Finanziari.

Spina dorsale oggettiva (DSP/spettro alla Ehlers + esponente di Hurst + stagionalita'),
segnali ciclici causali, e validazione anti-overfitting (in-sample 2/3, out-of-sample 1/3,
ipotesi nulla random-entry, walk-forward, semaforo). Ogni sezione include un riquadro
"Come si legge" per l'interpretazione corretta.

Deploy: GitHub -> Streamlit Cloud. La chiave EODHD va in Settings -> Secrets.
"""

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
from src.backtest import pct_returns
from src.validation import (simple_is_oos_split, locked_holdout_split,
                            evaluate_simple, walk_forward_evaluate)
from src import charts

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
        submitted = st.form_submit_button("▶️ Esegui / Aggiorna analisi",
                                          use_container_width=True, type="primary")

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

price = get_price_series(df)
split_pos, split_date = simple_is_oos_split(price, is_frac=2 / 3)
dev_end_pos, holdout_date = locked_holdout_split(price, holdout_frac) if holdout_frac > 0 \
    else (len(price), price.index[-1])

# KPI
returns_all = pct_returns(price)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prezzo", f"{price.iloc[-1]:,.2f}")
c2.metric("Osservazioni", f"{len(price):,}")
c3.metric("Periodo", f"{price.index[0].date()} → {price.index[-1].date()}")
c4.metric("Vol. annua", f"{returns_all.std() * np.sqrt(252) * 100:.1f}%")
c5.metric("Rendim. buy&hold", f"{(price.iloc[-1] / price.iloc[0] - 1) * 100:+.0f}%")
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

st.plotly_chart(charts.build_spectrum_chart(periods, power, thr, peaks, dom_period),
                use_container_width=True)
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
    st.dataframe(peaks_df, use_container_width=True, hide_index=True)

st.divider()

# ===================================================
# SEZIONE 2 — CICLO ISOLATO + SCALOGRAMMA
# ===================================================
st.header("2 · Il ciclo isolato e la sua deriva nel tempo")
osc = cycle_oscillator(price, dom_period, bandwidth)

st.plotly_chart(
    charts.build_price_cycle_chart(price, osc, split_date, holdout_date, ticker),
    use_container_width=True)
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
                    use_container_width=True)
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
st.header("3 · Dove siamo nel ciclo, adesso")
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
    st.plotly_chart(charts.build_phase_gauge(osc_last, state_label), use_container_width=True)
with colB:
    st.markdown(f"""
    **Lettura operativa attuale ({price.index[-1].date()}):**

    - Posizione nel ciclo: **{osc_last:+.2f}** → *{base_state}*
    - Direzione: **{'in salita ⬆️' if rising else 'in discesa ⬇️'}**

    Il cycle-timing suggerisce di essere **lunghi nella fase di salita** (dal minimo verso il
    massimo) e **fuori/corti nella discesa**. Questa e' l'*indicazione teorica* del ciclo:
    diventa un **segnale operativo** solo se supera la validazione più in basso.
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

st.plotly_chart(charts.build_hurst_chart(price, hurst), use_container_width=True)
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

cS1, cS2 = st.columns(2)
with cS1:
    st.plotly_chart(charts.build_seasonality_bars(dow, "Rendimento medio per giorno"),
                    use_container_width=True)
with cS2:
    st.plotly_chart(charts.build_seasonality_bars(moy, "Rendimento medio per mese"),
                    use_container_width=True)
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
banner(f"### {label} — {tagline}")
for r in ev["reasons"]:
    st.write("• " + r)

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
st.dataframe(metrics_table, use_container_width=True)
how_to_read(
    "confronta le tre righe. La strategia deve **reggere fuori campione**: se l'IS e' ottimo "
    "ma l'OOS crolla, e' overfitting. La riga **Buy & Hold OOS** e' il metro del *beta*: uno "
    "Sharpe OOS della strategia molto sotto il buy&hold significa che stai solo cavalcando il "
    "mercato con più rischio operativo.")

cE1, cE2 = st.columns([1.4, 1])
with cE1:
    st.plotly_chart(charts.build_equity_chart(ev["equity"],
                                              (1 + ev["bh_returns"]).cumprod(), split_date),
                    use_container_width=True)
    how_to_read(
        "equity a **base 100**. La linea verticale segna l'inizio dell'**OOS**: e' *lì* che "
        "conta la performance. A sinistra la strategia e' calibrata (facile sembrare brava); "
        "a destra e' il vero esame.")
with cE2:
    st.plotly_chart(charts.build_drawdown_chart(ev["equity"]), use_container_width=True)
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
        use_container_width=True)
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
    st.plotly_chart(charts.build_walkforward_bars(wf["folds"]), use_container_width=True)
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
banner(f"### {label} — {tagline}")

cV1, cV2, cV3 = st.columns(3)
cV1.metric("p-value vs null", f"{null['p_value_return']:.2f}",
           help="≤ soglia = timing significativo")
cV2.metric("Trade OOS", f"{ev['n_trades_oos']}",
           delta=f"min {min_trades}", delta_color="normal")
cV3.metric("Consistenza WF", f"{wf['consistency']*100:.0f}%" if wf["folds"] else "—")

if holdout_frac > 0:
    st.warning(f"🔒 **Holdout bloccato:** i dati dopo il **{holdout_date.date()}** "
               f"({holdout_frac*100:.0f}% finale) andrebbero guardati **una sola volta**, "
               "alla fine dello sviluppo. Ogni volta che modifichi i parametri guardando "
               "l'OOS, l'OOS diventa in-sample: l'holdout e' l'ultima difesa contro il "
               "data-snooping.")

st.caption("⚠️ Strumento di ricerca a scopo educativo. Nessun risultato passato garantisce "
           "performance future. Non e' consulenza finanziaria. · Kriterion Quant")
