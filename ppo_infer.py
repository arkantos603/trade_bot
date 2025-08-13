import argparse
import pandas as pd
from stable_baselines3 import PPO
from src.data import load_ohlc, add_returns
from src.envs.trading_env import TradingEnv, EnvConfig
from src.metrics import summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='BTC-USD')
    p.add_argument('--start', type=str, default='2024-01-01')
    p.add_argument('--end', type=str, default='2024-12-31')
    p.add_argument('--bank', type=float, default=10_000.0)
    p.add_argument('--stake', type=float, default=0.02)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--dd_penalty', type=float, default=1.0)
    p.add_argument('--turnover_penalty', type=float, default=0.0)
    p.add_argument('--hold_penalty', type=float, default=0.0)
    p.add_argument('--exec_on_next_bar', action='store_true')
    p.add_argument('--reward_mode', type=str, default='equity_diff', choices=['equity_diff','log_return'])
    args = p.parse_args()

    df = load_ohlc(args.ticker, start=args.start, end=args.end)
    df = add_returns(df)

    cfg = EnvConfig(
        stake_type='percent',
        stake_value=args.stake,
        fee_bps=5.0,
        slippage_bps=0.0,
        window_obs=30,
        exec_on_next_bar=args.exec_on_next_bar,
        reward_mode=args.reward_mode,
        dd_penalty=args.dd_penalty,
        turnover_penalty=args.turnover_penalty,
        hold_penalty=args.hold_penalty,
    )
    model = PPO.load(args.model_path)

    env = TradingEnv(df, args.bank, cfg)
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    logs_df, trades_df, signals_df = env.get_logs()
    equity = logs_df['equity']
    returns = equity.pct_change().fillna(0.0)
    print('Resumo:', summary(returns))
    equity.to_csv('ppo_equity.csv')
    trades_df.to_csv('ppo_trades.csv', index=False)
    signals_df.to_csv('ppo_signals.csv', index=False)
    print('Arquivos salvos: ppo_equity.csv, ppo_trades.csv, ppo_signals.csv')

if __name__ == '__main__':
    main()
