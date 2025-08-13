import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.data import load_ohlc, add_returns
from src.envs.trading_env import TradingEnv, EnvConfig

def make_env(df, bank, cfg):
    def _init():
        return TradingEnv(df, bank, cfg)
    return _init

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='BTC-USD')
    p.add_argument('--start', type=str, default='2018-01-01')
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--bank', type=float, default=10_000.0)
    p.add_argument('--stake', type=float, default=0.02)
    p.add_argument('--timesteps', type=int, default=200_000)
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
    env = DummyVecEnv([make_env(df, args.bank, cfg)])

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    os.makedirs('models', exist_ok=True)
    out = f'models/ppo_{args.ticker}.zip'
    model.save(out)
    print(f'Modelo salvo em: {out}')

if __name__ == "__main__":
    main()
