import pandas as pd
import numpy as np
import requests
import time
import random

# =======================
# Utils de datas/símbolos
# =======================

def _to_millis(date_str: str | None) -> int | None:
    if not date_str:
        return None
    ts = pd.to_datetime(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)

def _binance_interval(ival: str) -> str:
    """
    Converte intervalos comuns para o formato da Binance.
    Aceita: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w/1wk,1M/1mo
    """
    m = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d",
        "1wk": "1w", "1w": "1w",
        "1mo": "1M", "1M": "1M",
    }
    return m.get(ival, "1d")

def _to_binance_symbol(ticker: str) -> str:
    """
    Normaliza: 'BTC-USD','BTCUSD','BTC/USDT','BTCUSDT' -> 'BTCUSDT'
    Regra: se terminar com USD, mapeia para USDT.
    """
    if not ticker:
        raise ValueError("Ticker vazio.")
    t = ticker.strip().upper()
    t = t.replace("-", "").replace("/", "")
    if t.endswith("USDT"):
        return t
    if t.endswith("USD"):
        return t[:-3] + "USDT"
    # sem sufixo → assume USDT
    return t + "USDT"

# =======================
# Downloader Binance
# =======================

def _req_with_backoff(url: str, params: dict, max_retries: int = 6, base: float = 1.6, max_sleep: float = 20.0):
    """
    GET com backoff para lidar com HTTP 429/5xx.
    """
    last_exc = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429 or "too many requests" in r.text.lower():
                raise requests.HTTPError(f"429 Too Many Requests: {r.text}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            # backoff exponencial com jitter
            sleep_s = min(max_sleep, (base ** i) + random.uniform(0, 1.0))
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha desconhecida em requisição Binance.")

def load_ohlc_binance(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Baixa OHLC da API pública da Binance (sem API key).
    - Símbolos em USDT: mapeia 'BTC-USD' → 'BTCUSDT'
    - 'end' tratado como inclusivo (+1 dia)
    - Paginação até cobrir o período (limit até 1000 por chamada)
    Retorna DataFrame com: Open, High, Low, Close, Volume e índice datetime.
    """
    symbol = _to_binance_symbol(ticker)
    bint = _binance_interval(interval)

    start_ms = _to_millis(start)
    end_ms = None
    if end:
        end_inc = (pd.to_datetime(end) + pd.Timedelta(days=1))
        if end_inc.tzinfo is None:
            end_inc = end_inc.tz_localize("UTC")
        end_ms = int(end_inc.timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    curr = start_ms

    while True:
        params = {"symbol": symbol, "interval": bint, "limit": limit}
        if curr is not None:
            params["startTime"] = curr
        if end_ms is not None:
            params["endTime"] = end_ms

        chunk = _req_with_backoff(url, params)
        if not chunk:
            break
        all_rows.extend(chunk)

        last_open = chunk[-1][0]  # open_time em ms
        next_start = last_open + 1

        # sai se página não completa ou já passou do fim
        if len(chunk) < limit:
            break
        if end_ms is not None and next_start >= end_ms:
            break

        curr = next_start

        # guarda-chuva: evita loops muito longos (janela gigante em 1m)
        if len(all_rows) > 1_000_000:
            break

    if not all_rows:
        raise ValueError(f"Binance: nenhum dado retornado para {symbol} no período.")

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

# Mantemos o mesmo nome que o app usa
def load_ohlc_safe(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Versão 'safe' que chama somente a Binance.
    Mantém assinatura compatível com o app.
    """
    return load_ohlc_binance(ticker, start=start, end=end, interval=interval)

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Close" not in out.columns:
        raise KeyError(f"Coluna 'Close' não encontrada. Colunas: {list(out.columns)}")
    out["Return"] = out["Close"].pct_change().fillna(0.0)
    out["LogReturn"] = np.log1p(out["Return"]).fillna(0.0)
    return out
