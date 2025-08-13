from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .indicators import rsi
from .metrics import equity_curve, summary

@dataclass
class StrategyConfig:
    # --- Sinais (exemplo: RSI) ---
    rsi_period: int = 14
    entry_rsi_op: str = "<="      # "<=", "<", ">=", ">", "==", "!="
    entry_rsi_value: float = 30.0
    exit_rsi_enabled: bool = True
    exit_rsi_op: str = ">="       # "<=", "<", ">=", ">", "==", "!="
    exit_rsi_value: float = 70.0
    take_profit: Optional[float] = None  # 0.02 = 2%
    stop_loss: Optional[float] = None    # 0.01 = 1%

    # --- Tamanho da ordem ---
    # 'percent' = fração do capital (0.02 = 2%); 'usd' = valor fixo; 'qty' = quantidade de unidades
    order_size_type: str = "percent"     # 'percent' | 'usd' | 'qty'
    order_size_value: float = 0.02
    pyramid: int = 1                     # nº máx. de camadas (1 = sem pirâmide)

    # --- Execução / Custos ---
    order_type: str = "market"           # 'market' | 'limit'
    limit_offset_ticks: int = 0          # deslocamento do preço (ticks) p/ ordens limite
    tick_size: float = 0.01              # tamanho do tick em preço

    # Comissão por lado (entrada e saída)
    commission_unit: str = "bps"         # 'bps' | 'percent' | 'usd'
    commission_value: float = 5.0        # valor na unidade escolhida

    # Slippage aplicado no preço de execução
    slippage_unit: str = "bps"           # 'bps' | 'ticks'
    slippage_value: float = 0.0

    # --- Campos de margem (visuais; não alavancam no baseline) ---
    margin_long_pct: float = 100.0
    margin_short_pct: float = 100.0

def _apply_slippage(price: float, side: str, unit: str, value: float, tick_size: float) -> float:
    if unit == "bps":
        adj = price * (value / 10000.0)
    elif unit == "ticks":
        adj = value * tick_size
    else:
        adj = 0.0
    return price + adj if side == "buy" else price - adj

def _fee(amount_notional: float, unit: str, value: float) -> float:
    if unit == "bps":
        return amount_notional * (value / 10000.0)
    if unit == "percent":
        return amount_notional * (value / 100.0)
    if unit == "usd":
        return float(value)
    return 0.0

def _limit_fill(row: pd.Series, side: str, price_ref: float, offset_ticks: int, tick_size: float) -> Optional[float]:
    # monta preço limite em torno do preço de referência do candle (Close)
    if side == "buy":
        limit_price = price_ref - offset_ticks * tick_size
        if row["Low"] <= limit_price <= row["High"]:
            return float(limit_price)
        return None
    else:  # sell
        limit_price = price_ref + offset_ticks * tick_size
        if row["Low"] <= limit_price <= row["High"]:
            return float(limit_price)
        return None

def _entry_condition(rsi_val: Optional[float], op: str, thresh: float) -> bool:
    if rsi_val is None or np.isnan(rsi_val):
        return False
    if op == "<=": return rsi_val <= thresh
    if op == "<":  return rsi_val <  thresh
    if op == ">=": return rsi_val >= thresh
    if op == ">":  return rsi_val >  thresh
    if op == "==": return rsi_val == thresh
    if op == "!=": return rsi_val != thresh
    return False

def _exit_condition(rsi_val: Optional[float], op: str, thresh: float) -> bool:
    if rsi_val is None or np.isnan(rsi_val):
        return False
    if op == ">=": return rsi_val >= thresh
    if op == ">":  return rsi_val >  thresh
    if op == "<=": return rsi_val <= thresh
    if op == "<":  return rsi_val <  thresh
    if op == "==": return rsi_val == thresh
    if op == "!=": return rsi_val != thresh
    return False

def simulate_rule(df: pd.DataFrame, bank: float, cfg: StrategyConfig) -> Dict[str, Any]:
    data = df.copy()
    data["RSI"] = rsi(data["Close"], cfg.rsi_period)

    # Estado (permite piramidar)
    layers = 0
    qty_total = 0.0
    avg_price = None
    bank_balance = float(bank)

    equity_list = []
    trades = []
    signals = []

    # Acumuladores para a seção Desempenho
    fees_paid = 0.0
    max_qty_held = 0.0

    for idx, row in data.iterrows():
        price = float(row["Close"])
        rsi_val = float(row["RSI"]) if not pd.isna(row["RSI"]) else None

        # Equity = caixa + posição a mercado
        pos_value = 0.0 if qty_total == 0 else qty_total * price
        equity_t = bank_balance + pos_value
        equity_list.append((idx, equity_t))

        # -------- ENTRADA (com pirâmide) --------
        can_enter_more = (layers < int(cfg.pyramid)) or (qty_total == 0 and int(cfg.pyramid) >= 1)
        if can_enter_more and _entry_condition(rsi_val, cfg.entry_rsi_op, cfg.entry_rsi_value):
            # preço executado (market ou limit se tocar)
            exec_price = None
            if cfg.order_type == "limit":
                exec_price = _limit_fill(row, "buy", price, int(cfg.limit_offset_ticks), float(cfg.tick_size))
            else:
                exec_price = price

            if exec_price is not None:
                buy_px = _apply_slippage(exec_price, "buy", cfg.slippage_unit, float(cfg.slippage_value), float(cfg.tick_size))

                # tamanho da ordem (percent/usd/qty)
                if cfg.order_size_type == "percent":
                    stake_cash = bank_balance * float(cfg.order_size_value)
                    qty = 0.0 if buy_px == 0 else stake_cash / buy_px
                elif cfg.order_size_type == "usd":
                    stake_cash = float(cfg.order_size_value)
                    qty = 0.0 if buy_px == 0 else stake_cash / buy_px
                else:  # 'qty'
                    qty = float(cfg.order_size_value)
                    stake_cash = qty * buy_px

                # comissão de entrada
                fee_in = _fee(stake_cash, cfg.commission_unit, float(cfg.commission_value))

                if stake_cash + fee_in <= bank_balance and qty > 0:
                    bank_balance -= (stake_cash + fee_in)
                    fees_paid += float(fee_in)                     # acumula comissão
                    # atualiza preço médio e camadas
                    if qty_total == 0:
                        avg_price = buy_px
                        qty_total = qty
                    else:
                        new_notional = qty_total * avg_price + qty * buy_px
                        qty_total += qty
                        avg_price = new_notional / max(qty_total, 1e-12)
                    layers += 1
                    max_qty_held = max(max_qty_held, float(qty_total))  # acompanha máx. de contratos
                    signals.append({"time": idx, "type": "buy", "price": buy_px})

        # -------- SAÍDA (fecha tudo) --------
        if qty_total > 0:
            pnl_pct = (price - avg_price) / avg_price
            exit_cond = False
            reason = ""

            if cfg.take_profit is not None and pnl_pct >= cfg.take_profit:
                exit_cond, reason = True, "TP"
            if cfg.stop_loss is not None and pnl_pct <= -abs(cfg.stop_loss):
                exit_cond, reason = True, "SL"
            if cfg.exit_rsi_enabled and _exit_condition(rsi_val, cfg.exit_rsi_op, cfg.exit_rsi_value) and not exit_cond:
                exit_cond, reason = True, "RSI"

            if exit_cond:
                exec_price = None
                if cfg.order_type == "limit":
                    exec_price = _limit_fill(row, "sell", price, int(cfg.limit_offset_ticks), float(cfg.tick_size))
                else:
                    exec_price = price

                if exec_price is not None:
                    sell_px = _apply_slippage(exec_price, "sell", cfg.slippage_unit, float(cfg.slippage_value), float(cfg.tick_size))
                    gross = qty_total * sell_px
                    fee_out = _fee(gross, cfg.commission_unit, float(cfg.commission_value))
                    bank_balance += (gross - fee_out)
                    fees_paid += float(fee_out)                   # acumula comissão

                    trades.append({
                        "entry_time": idx,                         # (simplificado; por camadas seria 1 trade por layer)
                        "entry_price": float(avg_price),
                        "exit_time": idx,
                        "exit_price": float(sell_px),
                        "qty": float(qty_total),
                        # pnl_cash: líquido na saída (fee de entrada já foi debitado do caixa)
                        "pnl_cash": float((sell_px - avg_price) * qty_total - fee_out),
                        # % bruto (sem fees)
                        "pnl_pct_trade": float((sell_px - avg_price) / avg_price),
                        "reason": reason,
                    })
                    qty_total = 0.0
                    avg_price = None
                    layers = 0
                    signals.append({"time": idx, "type": "sell", "price": sell_px})

    # marcação a mercado no final
    open_pnl_cash = 0.0
    if qty_total > 0:
        last_price = float(data["Close"].iloc[-1])
        pos_value = qty_total * last_price
        equity_list.append((data.index[-1], bank_balance + pos_value))
        open_pnl_cash = float((last_price - avg_price) * qty_total)  # sem fee de saída

    equity = pd.Series({t: v for t, v in equity_list}).sort_index()
    strat_returns = equity.pct_change().fillna(0.0)
    stats = summary(strat_returns)

    # agrega métricas brutas para "Desempenho"
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        gross_cash = (trades_df["exit_price"] - trades_df["entry_price"]) * trades_df["qty"]
        gross_profit = float(gross_cash.clip(lower=0).sum())
        gross_loss = float((-gross_cash.clip(upper=0)).sum())
    else:
        gross_profit = 0.0
        gross_loss = 0.0

    perf = {
        "fees_paid": float(fees_paid),
        "open_pnl_cash": float(open_pnl_cash),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "max_qty_held": float(max_qty_held),
    }

    return {
        "equity": equity,
        "returns": strat_returns,
        "trades": trades_df,
        "signals": pd.DataFrame(signals),
        "final_bank": float(equity.iloc[-1]) if len(equity) else float(bank),
        "initial_bank": float(equity.iloc[0]) if len(equity) else float(bank),
        "stats": stats,
        "perf": perf,
    }
