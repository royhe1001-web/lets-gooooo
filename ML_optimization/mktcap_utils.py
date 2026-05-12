#!/usr/bin/env python3
"""Shared market cap lookup utility for ML pipeline."""

import os, pickle, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MktCap_PATH = os.path.join(BASE, 'output', 'monthly_mktcap.csv')
MktCap_CACHE = os.path.join(BASE, 'output', 'mktcap_lookup.pkl')

_lookup_cache = None


def load_mktcap():
    """Load monthly market cap snapshot. Returns DataFrame with ym column added."""
    df = pd.read_csv(MktCap_PATH)
    df['ym'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m')
    df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    return df


def build_mktcap_lookup(mktcap_df=None, min_cap=500, max_cap=1000):
    """Build (ym, symbol) -> mv_yi lookup dict. Uses pickle cache for speed."""
    global _lookup_cache
    if _lookup_cache is not None:
        return _lookup_cache

    # Try pickle cache first (fastest)
    if os.path.exists(MktCap_CACHE):
        with open(MktCap_CACHE, 'rb') as f:
            _lookup_cache = pickle.load(f)
        return _lookup_cache

    # Fallback: build from CSV
    if mktcap_df is None:
        mktcap_df = load_mktcap()
    lookup = {}
    for _, r in mktcap_df.iterrows():
        lookup[(r['ym'], r['symbol'])] = r['mv_yi']
    _lookup_cache = lookup
    return lookup


def get_mktcap(symbol, date, mktcap_lookup):
    """Get market cap (亿) for a symbol at a given date. Returns None if not found."""
    sym = str(symbol).zfill(6)
    ym = pd.Timestamp(date).strftime('%Y-%m')
    return mktcap_lookup.get((ym, sym))


def is_in_mktcap_range(symbol, date, mktcap_lookup, min_cap=500, max_cap=1000):
    """Check if a stock is within market cap range at a given date."""
    mv = get_mktcap(symbol, date, mktcap_lookup)
    if mv is None:
        return False
    return min_cap <= mv <= max_cap


def get_mktcap_universe(mktcap_lookup, ym, min_cap=500, max_cap=1000, ym_end=None):
    """Get set of stock codes within market cap range.

    If ym_end is None: single month lookup (fast for one month)
    If ym_end is provided: returns codes that were in range during ANY month in [ym, ym_end]
    """
    codes = set()
    if ym_end is None:
        for (m, sym), mv in mktcap_lookup.items():
            if m == ym and min_cap <= mv <= max_cap:
                codes.add(sym)
    else:
        for (m, sym), mv in mktcap_lookup.items():
            if ym <= m <= ym_end and min_cap <= mv <= max_cap:
                codes.add(sym)
    return codes
