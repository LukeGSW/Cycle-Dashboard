"""
registry.py — Registro dei parametri VALIDATI, associati al ticker (persistenza git-based).

Modello di persistenza (compatibile con Streamlit Cloud, filesystem effimero):
    - L'app LEGGE `presets.json` (nella root del repo) all'avvio -> richiamo automatico.
    - Al SALVATAGGIO (allo sblocco dell'esame holdout) l'app aggiorna il registro in
      sessione e offre il JSON aggiornato in download: l'utente lo COMMITTA su GitHub.
      La cronologia git diventa l'audit trail: ogni modifica e' un commit tracciabile.

Anti-imbroglio: ogni record porta un `config_hash` (fingerprint dei parametri) e la data di
blocco. Una config bloccata NON si modifica finche' non si esegue un Reset esplicito, che
archivia il vecchio record in `history`.
"""

from __future__ import annotations

import hashlib
import json
import os

# Parametri che definiscono univocamente una strategia (ordine stabile per l'hash)
PARAM_KEYS = [
    "start_date", "end_date", "min_p", "max_p", "hp_period", "bandwidth", "mode",
    "use_regime", "hurst_window", "max_hurst", "use_season", "cost_bps",
    "holdout_frac", "min_trades", "alpha", "min_sharpe_frac", "n_sims", "n_folds",
]

# Percorso robusto: root del repo (una cartella sopra src/), indipendente dalla CWD.
REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets.json")


def load_registry_file(path: str = REGISTRY_PATH) -> dict:
    """Carica il registro dal file JSON del repo. Ritorna {} se assente o illeggibile."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def config_hash(params: dict) -> str:
    """Fingerprint deterministico dei parametri (per tamper-evidence)."""
    payload = json.dumps({k: params.get(k) for k in PARAM_KEYS},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def make_record(params: dict, frozen: dict, holdout_result: dict, now_iso: str,
                old_record: dict | None = None) -> dict:
    """
    Costruisce un record di config validata. Se `old_record` esiste (sovrascrittura dopo
    Reset), lo archivia in `history`.

    Args:
        params:         dict dei parametri (verranno filtrati su PARAM_KEYS)
        frozen:         {dom_period, favorable_months} congelati sullo sviluppo
        holdout_result: {verdict, total_return, sharpe, p_value, n_trades, ...}
        now_iso:        timestamp ISO (l'app passa datetime.now().isoformat())
        old_record:     record precedente da archiviare, se presente

    Returns:
        dict record pronto per il registro.
    """
    rec = {
        "locked_at": now_iso,
        "params": {k: params.get(k) for k in PARAM_KEYS},
        "frozen": frozen,
        "holdout_result": holdout_result,
    }
    rec["config_hash"] = config_hash(rec["params"])

    history = []
    if old_record is not None:
        history = list(old_record.get("history", []))
        prior = {k: old_record.get(k) for k in
                 ("locked_at", "params", "frozen", "holdout_result", "config_hash")}
        history.append(prior)
    rec["history"] = history
    return rec


def registry_to_json(registry: dict) -> str:
    """Serializza il registro in JSON leggibile (da scaricare e committare)."""
    return json.dumps(registry, indent=2, ensure_ascii=False, default=str, sort_keys=True)
