"""
validation.py — Il cuore anti-overfitting: split, walk-forward, ipotesi nulla, semaforo.

Disciplina implementata (concordata in fase di design):
    1. Split IN-SAMPLE 2/3 vs OUT-OF-SAMPLE 1/3 (parametri clementi, non restrittivi).
    2. Ipotesi NULLA random-entry: si generano molte strategie con lo STESSO numero di
       trade e la stessa distribuzione di holding, con ingressi CASUALI, testate sullo
       stesso prezzo reale. Battere questo null significa che il TIMING del ciclo aggiunge
       valore oltre alla semplice frequenza operativa e alla deriva del mercato (beta).
    3. Confronto risk-adjusted vs buy & hold (non solo equity grezza: non basta cavalcare
       il beta).
    4. Numero minimo di trade OOS (sotto soglia nessuna statistica e' affidabile).
    5. Walk-forward ancorato: il periodo dominante viene RI-STIMATO su ogni train,
       verificando la stabilita' su piu' finestre OOS invece che su una sola.
    6. Holdout finale bloccato: da guardare una volta sola, mai per ottimizzare.

Semaforo:
    VERDE  = netto costi, batte il null (p<=alpha), risk-adj >= frazione del buy&hold,
             abbastanza trade.
    GIALLO = equity positiva ma fallisce uno dei criteri statistici -> paper trading.
    ROSSO  = equity negativa netta o peggio della mediana del null.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import (run_backtest, compute_metrics, buy_and_hold_returns,
                          count_trades)


# ============================================================
# SPLIT
# ============================================================
def simple_is_oos_split(price: pd.Series, is_frac: float = 2 / 3):
    """
    Split cronologico in-sample / out-of-sample.

    Returns:
        (split_pos, split_date): posizione intera e data di taglio. IS = [0, split_pos),
        OOS = [split_pos, fine).
    """
    n = len(price)
    split_pos = int(n * is_frac)
    return split_pos, price.index[split_pos]


def locked_holdout_split(price: pd.Series, holdout_frac: float = 0.2):
    """
    Isola un holdout finale 'bloccato' (ultima frazione della serie), da consultare una
    sola volta a fine sviluppo. Restituisce le date di confine.

    Returns:
        (develop_end_pos, holdout_start_date)
    """
    n = len(price)
    dev_end = int(n * (1 - holdout_frac))
    return dev_end, price.index[dev_end]


# ============================================================
# IPOTESI NULLA: RANDOM-ENTRY
# ============================================================
def random_entry_null(price: pd.Series, template_position: pd.Series,
                      n_sims: int = 400, cost_bps: float = 2.0,
                      seed: int = 42) -> dict:
    """
    Distribuzione nulla via strategie random-entry con trade e holding "clonati".

    Per ogni simulazione si RIPOSIZIONA casualmente lo stesso insieme di operazioni della
    strategia reale (stesso numero di run, STESSE durate, stessi segni), ridistribuendo in
    modo casuale i periodi di sosta. Cosi' l'ESPOSIZIONE TOTALE del null e' identica a quella
    della strategia (nessun overlap, nessun troncamento a fine serie): cambia solo il TIMING.
    Si fa il backtest sul PREZZO REALE. La distribuzione dei risultati e' l'ipotesi nulla
    "non c'e' timing, solo frequenza + deriva". Battere questo null = il timing aggiunge valore.

    Args:
        price:             prezzo del segmento su cui testare (tipicamente OOS)
        template_position: posizione reale della strategia sullo stesso segmento
        n_sims:            numero di simulazioni nulle
        cost_bps:          costi di transazione
        seed:              riproducibilita'

    Returns:
        dict con distribuzioni nulle e p-value one-sided della strategia reale.
    """
    idx = price.index
    n = len(price)
    pos = template_position.reindex(idx).fillna(0.0).values

    # Estrai i "run" di posizione non nulla (durata + segno)
    runs, signs = [], []
    i = 0
    while i < n:
        if pos[i] != 0.0:
            j = i
            while j < n and pos[j] == pos[i]:
                j += 1
            runs.append(j - i)
            signs.append(np.sign(pos[i]))
            i = j
        else:
            i += 1
    k = len(runs)

    strat_ret = run_backtest(price, template_position, cost_bps)["strat_returns"]
    strat_m = compute_metrics(strat_ret)
    strat_total = strat_m["total_return"]
    strat_sharpe = strat_m["sharpe"]

    if k == 0 or n < 3:
        return {
            "strat_total": strat_total, "strat_sharpe": strat_sharpe,
            "ret_dist": np.array([]), "sharpe_dist": np.array([]),
            "p_value_return": 1.0, "p_value_sharpe": 1.0,
            "null_median_return": np.nan, "null_p75_return": np.nan,
            "n_runs": 0,
        }

    rng = np.random.default_rng(seed)
    holding = np.array(runs, dtype=int)
    sign_arr = np.array(signs, dtype=float)
    total_in_market = int(holding.sum())
    gap_budget = max(n - total_in_market, 0)   # barre fuori mercato da ridistribuire
    n_slots = k + 1                            # gap: iniziale + (k-1) interni + finale
    interior = k - 1                           # i gap interni devono essere >=1 per non fondere i run
    ret_dist = np.empty(n_sims)
    sharpe_dist = np.empty(n_sims)

    for s in range(n_sims):
        # Ordine casuale dei run: preserva ESATTAMENTE il multiset di durate e segni
        order = rng.permutation(k)
        runs_perm = holding[order]
        signs_perm = sign_arr[order]

        # Ridistribuzione casuale del budget di sosta mantenendo l'esposizione totale.
        # sum(gaps) == gap_budget sempre -> sum(gaps)+sum(runs) == n (nessun troncamento).
        if gap_budget >= interior:
            extra = gap_budget - interior
            gaps = rng.multinomial(extra, np.full(n_slots, 1.0 / n_slots))
            if interior > 0:
                gaps[1:k] += 1                 # gap interni >=1 -> run separati -> trade count preservato
        else:
            gaps = rng.multinomial(gap_budget, np.full(n_slots, 1.0 / n_slots))

        sim = np.zeros(n)
        pos = 0
        for i in range(k):
            pos += int(gaps[i])
            end = min(pos + int(runs_perm[i]), n)
            sim[pos:end] = signs_perm[i]
            pos = end
        m = compute_metrics(run_backtest(price, pd.Series(sim, index=idx),
                                         cost_bps)["strat_returns"])
        ret_dist[s] = m["total_return"]
        sharpe_dist[s] = m["sharpe"]

    return {
        "strat_total": strat_total,
        "strat_sharpe": strat_sharpe,
        "ret_dist": ret_dist,
        "sharpe_dist": sharpe_dist,
        "p_value_return": float(np.mean(ret_dist >= strat_total)),
        "p_value_sharpe": float(np.mean(sharpe_dist >= strat_sharpe)),
        "null_median_return": float(np.median(ret_dist)),
        "null_p75_return": float(np.percentile(ret_dist, 75)),
        "n_runs": k,
    }


# ============================================================
# SEMAFORO
# ============================================================
def traffic_light(oos_m: dict, bh_oos_m: dict, null: dict, n_trades_oos: int,
                  min_trades: int = 20, alpha: float = 0.25,
                  min_sharpe_frac: float = 0.5):
    """
    Regola di accettazione a semaforo. Vedi docstring del modulo.

    Returns:
        (verdict, reasons, checks): verdict in {'GREEN','YELLOW','RED'}.
    """
    oos_tr = oos_m["total_return"]
    null_med = null.get("null_median_return", np.nan)

    # --- ROSSO ---
    if not (oos_tr == oos_tr) or oos_tr <= 0:
        return "RED", ["Equity OOS non positiva al netto dei costi."], {"positive": False}
    if null_med == null_med and oos_tr <= null_med:
        return "RED", [f"Peggio della mediana del null random-entry ({null_med:+.1%})."], \
               {"positive": True, "beats_null_median": False}

    # --- Criteri per il verde ---
    beats_null = null["p_value_return"] <= alpha
    enough_trades = n_trades_oos >= min_trades
    bh_sharpe = bh_oos_m.get("sharpe", np.nan)
    oos_sharpe = oos_m.get("sharpe", np.nan)
    if not (bh_sharpe == bh_sharpe) or bh_sharpe <= 0:
        beats_beta = oos_sharpe > 0
    else:
        beats_beta = oos_sharpe >= min_sharpe_frac * bh_sharpe

    checks = {
        "positive": True,
        "beats_null": bool(beats_null),
        "enough_trades": bool(enough_trades),
        "beats_beta": bool(beats_beta),
        "p_value": null["p_value_return"],
        "n_trades": n_trades_oos,
    }

    reasons = []
    if not beats_null:
        reasons.append(f"Non batte il null random-entry (p={null['p_value_return']:.2f} > {alpha}).")
    if not enough_trades:
        reasons.append(f"Troppi pochi trade OOS ({n_trades_oos} < {min_trades}).")
    if not beats_beta:
        reasons.append(f"Sharpe OOS {oos_sharpe:.2f} < {min_sharpe_frac:.0%} del buy&hold ({bh_sharpe:.2f}).")

    if beats_null and enough_trades and beats_beta:
        return "GREEN", ["Supera tutti i criteri OOS: segnali futuri accettabili."], checks
    return "YELLOW", reasons, checks


# ============================================================
# VALUTAZIONE COMPLETA — SPLIT SEMPLICE
# ============================================================
def evaluate_simple(price: pd.Series, position: pd.Series, is_frac: float = 2 / 3,
                    cost_bps: float = 2.0, min_trades: int = 20, alpha: float = 0.25,
                    min_sharpe_frac: float = 0.5, n_sims: int = 400,
                    seed: int = 42) -> dict:
    """
    Valutazione IS/OOS completa con null e semaforo.

    Args:
        price:           Serie di prezzi (intero campione)
        position:        posizione target sull'intero campione (NON shiftata)
        is_frac:         frazione in-sample (2/3 default)
        cost_bps:        costi di transazione (bps per unita' di turnover)
        min_trades:      soglia minima di trade OOS per il verde
        alpha:           soglia p-value vs null (0.25 = clemente)
        min_sharpe_frac: frazione minima dello Sharpe buy&hold OOS
        n_sims:          simulazioni nulle
        seed:            riproducibilita'

    Returns:
        dict completo con metriche IS/OOS, benchmark, null, semaforo ed equity.
    """
    split_pos, split_date = simple_is_oos_split(price, is_frac)
    bt = run_backtest(price, position, cost_bps)
    ret = bt["strat_returns"]

    is_metrics = compute_metrics(ret.iloc[:split_pos])
    oos_metrics = compute_metrics(ret.iloc[split_pos:])

    bh = buy_and_hold_returns(price)
    bh_is_metrics = compute_metrics(bh.iloc[:split_pos])
    bh_oos_metrics = compute_metrics(bh.iloc[split_pos:])

    price_oos = price.iloc[split_pos:]
    pos_oos = position.iloc[split_pos:]
    n_trades_oos = count_trades(pos_oos)

    null = random_entry_null(price_oos, pos_oos, n_sims=n_sims,
                             cost_bps=cost_bps, seed=seed)
    verdict, reasons, checks = traffic_light(
        oos_metrics, bh_oos_metrics, null, n_trades_oos,
        min_trades=min_trades, alpha=alpha, min_sharpe_frac=min_sharpe_frac)

    return {
        "split_pos": split_pos,
        "split_date": split_date,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "bh_is_metrics": bh_is_metrics,
        "bh_oos_metrics": bh_oos_metrics,
        "n_trades_oos": n_trades_oos,
        "null": null,
        "verdict": verdict,
        "reasons": reasons,
        "checks": checks,
        "equity": bt["equity"],
        "strat_returns": ret,
        "bh_returns": bh,
    }


# ============================================================
# WALK-FORWARD ANCORATO
# ============================================================
def walk_forward_evaluate(price: pd.Series, signal_builder: Callable[[int], pd.Series],
                          n_folds: int = 4, min_train_frac: float = 0.5,
                          cost_bps: float = 2.0) -> dict:
    """
    Walk-forward ancorato (finestra di train espandente). Per ogni fold il segnale viene
    ri-calibrato usando SOLO i dati fino all'inizio del fold (nessun look-ahead).

    Args:
        price:          Serie di prezzi
        signal_builder: funzione (train_end_pos:int) -> posizione sull'intero indice,
                        calibrata solo su price[:train_end_pos]. E' il chiamante a
                        implementare la ricalibrazione (es. ri-stima del periodo dominante).
        n_folds:        numero di finestre OOS consecutive
        min_train_frac: frazione iniziale di train prima del primo fold
        cost_bps:       costi

    Returns:
        dict con 'folds' (lista di metriche per fold) e 'consistency' (frazione di fold
        con rendimento OOS positivo).
    """
    n = len(price)
    start = int(n * min_train_frac)
    chunk = max((n - start) // n_folds, 1)

    folds = []
    positives = 0
    for f in range(n_folds):
        test_start = start + f * chunk
        test_end = (start + (f + 1) * chunk) if f < n_folds - 1 else n
        if test_start >= n:
            break
        pos = signal_builder(test_start)
        bt = run_backtest(price, pos, cost_bps)
        oos_ret = bt["strat_returns"].iloc[test_start:test_end]
        m = compute_metrics(oos_ret)
        is_pos = m["total_return"] > 0
        positives += int(is_pos)
        folds.append({
            "fold": f + 1,
            "train_end": price.index[test_start - 1],
            "test_start": price.index[test_start],
            "test_end": price.index[test_end - 1],
            "total_return": m["total_return"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "positive": bool(is_pos),
        })

    consistency = positives / len(folds) if folds else 0.0
    return {"folds": folds, "consistency": consistency}
