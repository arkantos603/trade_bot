import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import requests

_FIELDS = {"open", "high", "low", "close", "adj close", "volume"}

def _standardize_field_names(cols) -> list[str]:
    out = []
    for c in cols:
        s = str(c).strip()
        low = s.lower()
        if low in _FIELDS:
            if low == "adj close":
                out.append("Adj Close")
            else:
                out.append(s.title())
        else:
            out.append(s)
    return out

def load_ohlc(
    ticker: str,
    start: str = None,
    end: str = None,
    interval: str = "1d",
    period: str | None = None,
) -> pd.DataFrame:
    if period:
        df = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True
        )
    else:
        df = yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, threads=True
        )

    if df is None or len(df) == 0:
        raise ValueError(f"Nenhum dado retornado para '{ticker}'.")

    # Colunas MultiIndex (alguns tickers)
    if isinstance(df.columns, pd.MultiIndex):
        sliced = None
        for lvl in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(lvl):
                sliced = df.xs(ticker, axis=1, level=lvl, drop_level=True)
                break
        df = sliced if sliced is not None else df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    df.columns = _standardize_field_names(df.columns)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index().dropna(how="all")

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if len(needed - set(df.columns)) > 0:
        raise KeyError(f"Colunas essenciais ausentes após normalização. Colunas: {list(df.columns)}")

    return df

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns:
        raise KeyError(f"Coluna 'Close' não encontrada. Colunas: {list(out.columns)}")
    out["Return"] = out["Close"].pct_change().fillna(0.0)
    out["LogReturn"] = np.log1p(out["Return"]).fillna(0.0)
    return out


# ----------------- Helpers de ticker e datas -----------------
def _normalize_ticker_candidates(ticker: str) -> list[str]:
    t = (ticker or "").strip()
    if not t:
        return []
    u = t.upper()
    cands = [t]  # como veio

    # Mapeamentos comuns (pode ampliar)
    crypto_map = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "ADAUSDT": "ADA-USD",
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
    }
    if u in crypto_map:
        cands.append(crypto_map[u])

    # B3: PETR4 -> PETR4.SA (se não tiver sufixo)
    if "." not in t and (t.endswith(("3", "4", "11")) and len(t) <= 6):
        cands.append(f"{t}.SA")

    # Forex: EURUSD -> EURUSD=X (6 letras)
    if len(u) == 6 and u.isalpha():
        cands.append(f"{u}=X")

    # remove duplicatas preservando ordem
    seen, out = set(), []
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _is_crypto_usd_symbol(ticker: str) -> bool:
    """Heurística simples: BTC-USD, ETH-USD, SOL-USD, etc."""
    if not ticker:
        return False
    u = ticker.upper()
    return u.endswith("-USD") or u.endswith("USDT") or u in {"BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD"}

def _to_millis(date_str: str | None) -> int | None:
    if not date_str:
        return None
    ts = pd.to_datetime(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)

def _binance_interval(ival: str) -> str:
    """Mapeia intervalos '1d','1h','1m','1wk','1mo' -> Binance."""
    m = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d",
        "1wk": "1w", "1w": "1w",
        "1mo": "1M", "1M": "1M",
    }
    return m.get(ival, "1d")


# ----------------- Fallback: Binance (cripto) -----------------
def load_ohlc_binance_crypto_usd(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Baixa OHLC de cripto USD pela API pública da Binance (sem API key).
    Converte 'BTC-USD' -> 'BTCUSDT'. Pagina até cobrir o período.
    """
    # Mapeia 'BTC-USD'/'BTCUSD'/'BTCUSDT' -> 'BTCUSDT'
    u = (ticker or "").upper().replace("-", "")
    if u.endswith("USD"):
        sym = u[:-3] + "USDT"
    elif u.endswith("USDT"):
        sym = u
    else:
        # fallback conservador
        sym = u + "USDT"

    bint = _binance_interval(interval)
    start_ms = _to_millis(start)
    # torna 'end' inclusivo (+1 dia), parecido com o Yahoo
    end_inc = None
    if end:
        end_inc = (pd.to_datetime(end) + pd.Timedelta(days=1))
        if end_inc.tzinfo is None:
            end_inc = end_inc.tz_localize("UTC")
        end_ms = int(end_inc.timestamp() * 1000)
    else:
        end_ms = None

    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    curr = start_ms
    while True:
        params = {"symbol": sym, "interval": bint, "limit": limit}
        if curr is not None:
            params["startTime"] = curr
        if end_ms is not None:
            params["endTime"] = end_ms

        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        # paginação: próximo início = último open_time + 1 ms
        last_open = chunk[-1][0]
        next_start = last_open + 1
        # se não recebemos página cheia, paramos
        if len(chunk) < limit:
            break
        # se end_ms definido e próximo início já passa do fim, paramos
        if end_ms is not None and next_start >= end_ms:
            break
        curr = next_start

    if not all_rows:
        raise ValueError(f"Binance: nenhum dado retornado para {sym} no período.")

    # Converte para DataFrame
    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "n_trades",
        "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated()].sort_index()

    return df


# ----------------- Loader seguro com Yahoo + Fallback Binance -----------------
def load_ohlc_safe(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Baixa OHLC com:
      - Yahoo Finance (com backoff e fallbacks)
      - Fallback automático para Binance se o símbolo for cripto USD
    Trata HTTP 429 (Too Many Requests) e outros erros de rede.
    """
    import time, random

    candidates = _normalize_ticker_candidates(ticker)
    if not candidates:
        raise ValueError("Ticker vazio.")

    # Normaliza datas e torna 'end' inclusivo (Yahoo trata end como exclusivo)
    start_s = pd.to_datetime(start).strftime("%Y-%m-%d") if start else None
    end_inclusive = None
    if end:
        end_inclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _with_backoff(fn, max_retries=5, base=1.5, max_sleep=30):
        last_exc = None
        for i in range(max_retries):
            try:
                df = fn()
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                    sleep_s = min(max_sleep, (base ** i) + random.uniform(0, 1.0))
                    time.sleep(sleep_s)
                    continue
                # outros erros: tenta próximo fallback
        if last_exc:
            raise last_exc
        raise ValueError("Falha desconhecida ao baixar dados.")

    # ---------- 1) Yahoo (start/end + threads=True)
    for tk in candidates:
        try:
            return _with_backoff(lambda: load_ohlc(tk, start=start_s, end=end_inclusive, interval=interval))
        except Exception:
            pass

    # ---------- 2) Yahoo (start/end com threads=False)
    for tk in candidates:
        try:
            def _dl():
                return yf.download(
                    tk, start=start_s, end=end_inclusive, interval=interval,
                    auto_adjust=True, progress=False, threads=False
                )
            df = _with_backoff(_dl)
            if df is not None and len(df) > 0:
                # reaplica normalização via load_ohlc
                return load_ohlc(tk, start=start_s, end=end_inclusive, interval=interval)
        except Exception:
            pass

    # ---------- 3) Yahoo .history()
    for tk in candidates:
        try:
            def _hist():
                t = yf.Ticker(tk)
                return t.history(start=start_s, end=end_inclusive, interval=interval, auto_adjust=True)
            df = _with_backoff(_hist)
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df.index)
                if "Adj Close" in df and "Close" not in df:
                    df["Close"] = df["Adj Close"]
                if "Volume" not in df:
                    df["Volume"] = 0.0
                df = df.sort_index()
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass

    # ---------- 4) Yahoo period='max'
    for tk in candidates:
        try:
            return _with_backoff(lambda: load_ohlc(tk, period="max", interval=interval))
        except Exception:
            pass

    # ---------- 5) Fallback: Binance (apenas para cripto USD)
    if _is_crypto_usd_symbol(ticker):
        try:
            return load_ohlc_binance_crypto_usd(ticker, start=start, end=end, interval=interval)
        except Exception as e:
            # deixa mensagem útil
            raise ValueError(f"Nenhum dado retornado do Yahoo e falha no fallback Binance: {e}")

    # Nada deu certo
    raise ValueError(f"Nenhum dado retornado. Tickers tentados: {candidates}")
