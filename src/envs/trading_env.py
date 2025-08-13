from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class EnvConfig:
    stake_type: str = "percent"
    stake_value: float = 0.02
    fee_bps: float = 5.0
    slippage_bps: float = 0.0
    window_obs: int = 30
    allow_short: bool = False
    exec_on_next_bar: bool = False

    reward_mode: str = "equity_diff"  # 'equity_diff' or 'log_return'
    dd_penalty: float = 1.0
    turnover_penalty: float = 0.0
    hold_penalty: float = 0.0

class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, bank: float, cfg: EnvConfig):
        super().__init__()
        self.df = df.copy()
        self.bank0 = float(bank)
        self.cfg = cfg

        self.returns = self.df["Close"].pct_change().fillna(0.0).values.astype(np.float32)
        rsi = self._rsi(self.df["Close"].values, 14).astype(np.float32)
        rsi = np.nan_to_num(rsi, nan=50.0)
        self.rsi = rsi

        self.window = cfg.window_obs
        obs_len = self.window * 2 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(3 if not cfg.allow_short else 5)

        self._reset_state()

    def _rsi(self, prices, period=14):
        p = np.asarray(prices, dtype=np.float32)
        delta = np.diff(p, prepend=p[0])
        gain = np.clip(delta, 0, None)
        loss = -np.clip(delta, None, 0)
        gain = pd.Series(gain).rolling(period).mean().values
        loss = pd.Series(loss).rolling(period).mean().values
        rs = gain / np.where(loss == 0, 1e-6, loss)
        return 100 - (100 / (1 + rs))

    def _price(self, t):
        return float(self.df["Close"].iloc[t])

    def _obs(self):
        sl = slice(self.t - self.window, self.t)
        ret_w = self.returns[sl]
        rsi_w = self.rsi[sl] / 100.0
        pos_flag = np.array([1.0 if self.qty > 0 else 0.0], dtype=np.float32)
        cash_ratio = np.array([self.bank / (self.bank0 + 1e-6)], dtype=np.float32)
        return np.concatenate([ret_w, rsi_w, pos_flag, cash_ratio]).astype(np.float32)

    def _fee_amt(self, cash_amt: float) -> float:
        return cash_amt * (self.cfg.fee_bps / 10000.0)

    def _reset_state(self):
        self.t = self.window
        self.bank = self.bank0
        self.qty = 0.0
        self.entry_price = None
        self.equity_prev = self.bank0
        self.peak_equity = self.bank0
        self.prev_action = 0
        self.pending_action = None

        self.logs = {"equity": [], "time": [], "action": [], "price": [], "bank": [], "qty": [], "trades": [], "signals": []}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        # Execução T+1 (se habilitada)
        if self.cfg.exec_on_next_bar and self.pending_action is not None:
            self._execute_action(self.pending_action)
            self.pending_action = None
        elif not self.cfg.exec_on_next_bar:
            self._execute_action(action)

        price = self._price(self.t)
        pos_val = 0.0 if self.qty == 0 else self.qty * price
        equity = self.bank + pos_val

        if self.cfg.reward_mode == "log_return":
            base_reward = float(np.log((equity + 1e-8) / (self.equity_prev + 1e-8)))
        else:
            base_reward = float(equity - self.equity_prev)

        dd_pen = 0.0
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (equity - self.peak_equity) / self.peak_equity
        prev_dd = (self.equity_prev - self.peak_equity) / self.peak_equity
        if dd < prev_dd:
            dd_pen = self.cfg.dd_penalty * abs(dd - prev_dd)

        turn_pen = self.cfg.turnover_penalty * (1.0 if action != self.prev_action else 0.0)
        hold_pen = self.cfg.hold_penalty * (1.0 if self.qty > 0 else 0.0)

        reward = base_reward - dd_pen - turn_pen - hold_pen

        self.logs["equity"].append(equity)
        self.logs["time"].append(self.df.index[self.t])
        self.logs["action"].append(int(action))
        self.logs["price"].append(price)
        self.logs["bank"].append(self.bank)
        self.logs["qty"].append(self.qty)

        if self.cfg.exec_on_next_bar:
            self.pending_action = action

        self.equity_prev = equity
        self.prev_action = action
        self.t += 1
        terminated = self.t >= len(self.df) - 1

        return self._obs(), reward, terminated, False, {"equity": equity}

    def _execute_action(self, action: int):
        price = self._price(self.t)
        if action == 1:  # Buy/open
            if self.qty == 0:
                stake_cash = self.bank * self.cfg.stake_value if self.cfg.stake_type == "percent" else self.cfg.stake_value
                if stake_cash > 0 and stake_cash <= self.bank:
                    buy_price = price * (1 + self.cfg.slippage_bps / 10000.0)
                    qty = stake_cash / buy_price
                    fee = self._fee_amt(stake_cash)
                    if stake_cash + fee <= self.bank:
                        self.bank -= (stake_cash + fee)
                        self.qty = qty
                        self.entry_price = buy_price
                        self.logs["signals"].append({"time": self.df.index[self.t], "type": "buy", "price": buy_price})
        elif action == 2:  # Close
            if self.qty > 0:
                gross = self.qty * price * (1 - self.cfg.slippage_bps / 10000.0)
                fee = self._fee_amt(gross)
                self.bank += (gross - fee)
                trade = {
                    "entry_time": self.df.index[self.t],
                    "entry_price": self.entry_price,
                    "exit_time": self.df.index[self.t],
                    "exit_price": price,
                    "qty": self.qty,
                    "pnl_cash": (price - self.entry_price) * self.qty - fee,
                    "pnl_pct_trade": (price - self.entry_price) / self.entry_price,
                    "reason": "agent",
                }
                self.logs["trades"].append(trade)
                self.qty = 0.0
                self.entry_price = None
                self.logs["signals"].append({"time": self.df.index[self.t], "type": "sell", "price": price})

    def get_logs(self):
        logs_df = pd.DataFrame({
            "time": self.logs["time"],
            "equity": self.logs["equity"],
            "action": self.logs["action"],
            "price": self.logs["price"],
            "bank": self.logs["bank"],
            "qty": self.logs["qty"],
        }).set_index("time")
        trades_df = pd.DataFrame(self.logs["trades"])
        signals_df = pd.DataFrame(self.logs["signals"])
        return logs_df, trades_df, signals_df
