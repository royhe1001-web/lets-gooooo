#!/usr/bin/env python3
"""Shared CSI 500 + CSI 1000 index constituent lookup for ML pipeline."""

import os, pickle, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_CACHE = os.path.join(BASE, 'output', 'index_constituent_lookup.pkl')

_lookup_cache = None


def build_index_lookup():
    """Load (ym, symbol) -> True dict for CSI500 + CSI1000 membership.

    Uses pickle cache (fast). Returns dict where key = (ym_str, symbol_6digit).
    """
    global _lookup_cache
    if _lookup_cache is not None:
        return _lookup_cache

    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, 'rb') as f:
            data = pickle.load(f)
        _lookup_cache = data['lookup']
        return _lookup_cache

    raise FileNotFoundError(f"Index cache not found: {INDEX_CACHE}. Run fetch first.")


def get_index_membership(symbol, date, index_lookup):
    """Check if a stock is in CSI500 or CSI1000 at a given date.

    Matches to the nearest available index snapshot (snapshot dates are
    spaced ~6 months apart).

    Returns True if the stock is in the index, False if not found or unknown.
    """
    sym = str(symbol).zfill(6)
    ym = pd.Timestamp(date).strftime('%Y-%m')
    return index_lookup.get((ym, sym), False)


def get_index_eligible_codes(index_lookup, ym_start='2018-01', ym_end='2026-05'):
    """Get set of all stock codes ever in CSI500 or CSI1000 during a date range."""
    codes = set()
    for (ym, sym) in index_lookup:
        if ym_start <= ym <= ym_end:
            codes.add(sym)
    return codes


def load_index_stats():
    """Return per-month stats dict (for reporting)."""
    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, 'rb') as f:
            data = pickle.load(f)
        return data.get('stats', {})
    return {}
