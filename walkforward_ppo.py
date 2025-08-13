import argparse
import os
import pandas as pd
from typing import List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.data import load_ohlc, add_returns
from src.envs.trading_env import TradingEnv, EnvConfig
from src.metrics import summary

def make_env(df, bank, cfg):
    def _init():
        return TradingEnv(df, bank, cfg)
    return _init

def split_walkforward(df: pd.DataFrame, train_days: int, test_days: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    # Return list of (train_df, test_df) non-overlapping, walk-forward windows.
    df = df.copy()
    n = len(df)
    i = 0
    windows = []
    while True:
        train_start = i
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > n:
            break
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        if len(train_df) > 0 and len(test_df) > 0:
            windows.append((train_df, test_df))
        i = train_end
    return windows

def run_fold(train_df, test_df, bank, stake, timesteps, cfg_kwargs):
    cfg = EnvConfig(
        stake_type='percent',
        stake_value=stake,
        fee_bps=cfg_kwargs.get('fee_bps', 5.0),
        slippage_bps=cfg_kwargs.get('slippage_bps', 0.0),
        window_obs=cfg_kwargs.get('window_obs', 30),
        exec_on_next_bar=cfg_kwargs.get('exec_on_next_bar', False),
        reward_mode=cfg_kwargs.get('reward_mode', 'equity_diff'),
        dd_penalty=cfg_kwargs.get('dd_penalty', 1.0),
        turnover_penalty=cfg_kwargs.get('turnover_penalty', 0.0),
        hold_penalty=cfg_kwargs.get('hold_penalty', 0.0),
    )
    env_train = DummyVecEnv([make_env(train_df, bank, cfg)])
    model = PPO('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)

    env_test = TradingEnv(test_df, bank, cfg)
    obs, _ = env_test.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(int(action))
        if terminated or truncated:
            break

    logs_df, trades_df, signals_df = env_test.get_logs()
    equity = logs_df['equity']
    returns = equity.pct_change().fillna(0.0)
    stats = summary(returns)

    return {'equity': equity, 'returns': returns, 'stats': stats, 'trades': trades_df, 'signals': signals_df}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='BTC-USD')
    p.add_argument('--start', type=str, default='2018-01-01')
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--bank', type=float, default=10_000.0)
    p.add_argument('--stake', type=float, default=0.02)
    p.add_argument('--train_days', type=int, default=504)
    p.add_argument('--test_days', type=int, default=21)
    p.add_argument('--timesteps', type=int, default=50_000)
    p.add_argument('--dd_penalty', type=float, default=1.0)
    p.add_argument('--turnover_penalty', type=float, default=0.0)
    p.add_argument('--hold_penalty', type=float, default=0.0)
    p.add_argument('--exec_on_next_bar', action='store_true')
    p.add_argument('--reward_mode', type=str, default='equity_diff', choices=['equity_diff','log_return'])
    args = p.parse_args()

    df = load_ohlc(args.ticker, start=args.start, end=args.end)
    df = add_returns(df)

    windows = split_walkforward(df, args.train_days, args.test_days)

    eq_all = []
    for i, (train_df, test_df) in enumerate(windows, 1):
        res = run_fold(
            train_df, test_df, args.bank, args.stake, args.timesteps,
            cfg_kwargs=dict(
                fee_bps=5.0,
                slippage_bps=0.0,
                window_obs=30,
                exec_on_next_bar=args.exec_on_next_bar,
                reward_mode=args.reward_mode,
                dd_penalty=args.dd_penalty,
                turnover_penalty=args.turnover_penalty,
                hold_penalty=args.hold_penalty,
            )
        )
        eq_all.append(res['equity'])
        print(f'Fold {i}: {res["stats"]}')

    equity_concat = pd.concat(eq_all).sort_index()
    returns_concat = equity_concat.pct_change().fillna(0.0)
    overall = summary(returns_concat)

    os.makedirs('wf_results', exist_ok=True)
    equity_concat.to_csv('wf_results/equity.csv')
    returns_concat.to_csv('wf_results/returns.csv')
    with open('wf_results/summary.txt', 'w', encoding='utf-8') as f:
        f.write(str(overall))

    print('=== Overall (walk-forward) ===')
    print(overall)

if __name__ == '__main__':
    main()
