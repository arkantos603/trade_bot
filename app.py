import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data import load_ohlc_safe, add_returns
from src.sim_rule import StrategyConfig, simulate_rule

# ---------------- Helpers da seção "Desempenho" ----------------
def _fmt_usd_pct(val_cash: float, pct_base: float) -> str:
    """Formata 'X USD (+Y%)' com base no capital inicial para o %."""
    if pct_base == 0:
        pct = 0.0
    else:
        pct = 100.0 * (val_cash / pct_base)
    sign = "+" if val_cash > 0 else ""
    return f"{val_cash:,.2f} USD ({sign}{pct:.2f}%)"

def _performance_table(df_price: pd.DataFrame, res: dict) -> pd.DataFrame:
    eq = res["equity"]
    trades = res["trades"]
    perf = res.get("perf", {})
    initial = float(res["initial_bank"])
    final = float(res["final_bank"])

    # Itens principais
    net_profit_cash = final - initial
    open_pnl_cash = float(perf.get("open_pnl_cash", 0.0))
    gross_profit = float(perf.get("gross_profit", 0.0))
    gross_loss = float(perf.get("gross_loss", 0.0))
    fees_paid = float(perf.get("fees_paid", 0.0))
    max_qty_held = float(perf.get("max_qty_held", 0.0))

    # Buy & Hold sobre o mesmo período
    p0 = float(df_price["Close"].iloc[0])
    p1 = float(df_price["Close"].iloc[-1])
    bh_final = initial * (p1 / p0) if p0 != 0 else initial
    bh_profit = bh_final - initial

    # Máx. run-up/drawdown em dinheiro
    runup_cash = float((eq - eq.cummin()).max())
    drawdown_cash = float((eq.cummax() - eq).max())

    linhas = {
        "L&P Aberto":                [_fmt_usd_pct(open_pnl_cash, initial), _fmt_usd_pct(open_pnl_cash, initial), _fmt_usd_pct(0.0, initial)],
        "Lucro líquido":             [_fmt_usd_pct(net_profit_cash, initial), _fmt_usd_pct(net_profit_cash, initial), _fmt_usd_pct(0.0, initial)],
        "Lucro Bruto":               [_fmt_usd_pct(gross_profit, initial), _fmt_usd_pct(gross_profit, initial), _fmt_usd_pct(0.0, initial)],
        "Prejuízo bruto":            [_fmt_usd_pct(gross_loss, initial), _fmt_usd_pct(gross_loss, initial), _fmt_usd_pct(0.0, initial)],
        "Comissão paga":             [f"{fees_paid:,.2f} USD", f"{fees_paid:,.2f} USD", "0,00 USD"],
        "Retorno do buy & hold":     [_fmt_usd_pct(bh_profit, initial), _fmt_usd_pct(bh_profit, initial), _fmt_usd_pct(0.0, initial)],
        "Máx. equity run-up":        [_fmt_usd_pct(runup_cash, initial), _fmt_usd_pct(runup_cash, initial), _fmt_usd_pct(0.0, initial)],
        "Máx. equity drawdown":      [_fmt_usd_pct(drawdown_cash, initial), _fmt_usd_pct(drawdown_cash, initial), _fmt_usd_pct(0.0, initial)],
        "Máx. de contratos detidos": [f"{max_qty_held:,.0f}", f"{max_qty_held:,.0f}", "0"],
    }
    return pd.DataFrame.from_dict(linhas, orient="index", columns=["Todos", "Viés de alta", "Viés de baixa"])

# Cache para não bater no Yahoo toda hora
@st.cache_data(show_spinner=False, ttl=1800)  # 30 min
def _fetch_prices(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = load_ohlc_safe(ticker, start=start, end=end)
    return add_returns(df)

# ---------------- UI ----------------
st.set_page_config(page_title="PPO Trading + Rule Simulator (PnL-first)", layout="wide")
st.title("PPO Trading + Simulador por Regras — Foco em Lucratividade")

with st.sidebar:
    st.header("Dados")
    ticker = st.text_input("Ticker (Yahoo Finance)", "BTC-USD")
    start = st.date_input("Início", pd.to_datetime("2018-01-01"))
    end = st.date_input("Fim", pd.to_datetime("2024-12-31"))

# Configurações iniciais
st.subheader("Configurações Iniciais (antes do teste)")
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    base_currency = st.text_input("Moeda de base", "USD")
    bank = st.number_input("Capital inicial", min_value=0.0, value=10000.0, step=100.0)
    order_size_value = st.number_input(
        "Tamanho da ordem (valor)",
        min_value=0.0, value=2.0, step=0.5,
        help='Se "% do capital", informe em %; se USD, valor absoluto; se Quantidade, nº de unidades.',
    )

with c2:
    order_size_type = st.selectbox("Unidade do tamanho da ordem", ["% do capital", "USD", "Quantidade"], index=0)
    pyramid = st.number_input("Pirâmide (camadas máx.)", min_value=1, value=1, step=1)
    commission_unit = st.selectbox("Comissão - unidade", ["%", "bps", "USD por lado"], index=1)

with c3:
    commission_value = st.number_input("Comissão - valor", min_value=0.0, value=5.0, step=0.5)
    slippage_unit = st.selectbox("Derrapagem - unidade", ["bps", "ticks"], index=0)
    slippage_value = st.number_input("Derrapagem - valor", min_value=0.0, value=0.0, step=0.5)

st.markdown("---")
st.subheader("Execução")
colA, colB, colC = st.columns(3)
with colA:
    order_type = st.selectbox("Tipo de ordem", ["market", "limit"], index=0)
with colB:
    limit_offset_ticks = st.number_input("Verificar preços p/ ordens limite (ticks)", min_value=0, value=0, step=1)
with colC:
    tick_size = st.number_input(
        "Tamanho do tick (em preço)", min_value=0.0, value=0.01, step=0.01,
        help="Ex.: ações US: 0.01; cripto: 0.1/0.01 conforme par/exchange.",
    )

# Sinais RSI / TP / SL
st.markdown("---")
st.subheader("Sinais RSI / TP / SL")
col1, col2, col3 = st.columns(3)
with col1:
    rsi_period = st.number_input("RSI período", 5, 50, 14)
    entry_rsi_value = st.number_input("RSI de Entrada (≤)", 0, 100, 30)
with col2:
    exit_rsi_enabled = st.checkbox("Usar RSI de Saída (≥)", True)
    exit_rsi_value = st.number_input("RSI de Saída (≥)", 0, 100, 70)
with col3:
    take_profit = st.number_input("Take Profit (%)", 0.0, 100.0, 2.0, step=0.1) / 100.0
    stop_loss = st.number_input("Stop Loss (%)", 0.0, 100.0, 1.0, step=0.1) / 100.0

# ---------------- Rodar ----------------
if st.button("Rodar Simulação"):
    with st.spinner("Carregando dados e simulando..."):
        # Carrega com fallback + cache
        try:
            df = _fetch_prices(ticker, str(start), str(end))
        except Exception as e:
            msg = str(e)
            if ("Too Many Requests" in msg) or ("Rate limited" in msg) or ("429" in msg):
                st.warning(
                    "O Yahoo Finance limitou as requisições desta máquina (HTTP 429). "
                    "Aguarde ~1–2 minutos e clique em **Rodar Simulação** novamente.\n\n"
                    "Dica: evite rodar várias vezes em sequência; o app faz cache por 30 min."
                )
            else:
                st.error(
                    "Não consegui baixar dados para esse símbolo/período.\n\n"
                    "Dicas de símbolo (Yahoo Finance):\n"
                    "- Cripto: BTC-USD, ETH-USD\n"
                    "- B3: PETR4.SA, VALE3.SA, BOVA11.SA\n"
                    "- Forex: EURUSD=X, USDJPY=X\n"
                    "- Futuros: ES=F, NQ=F\n\n"
                    f"Detalhes: {e}"
                )
            st.stop()

        # mapear unidades da UI -> config interna
        if order_size_type == "% do capital":
            order_size_type_internal = "percent"
            order_size_value_internal = order_size_value / 100.0
        elif order_size_type == "USD":
            order_size_type_internal = "usd"
            order_size_value_internal = order_size_value
        else:
            order_size_type_internal = "qty"
            order_size_value_internal = order_size_value

        if commission_unit == "%":
            commission_unit_internal = "percent"
        elif commission_unit == "USD por lado":
            commission_unit_internal = "usd"
        else:
            commission_unit_internal = "bps"

        cfg = StrategyConfig(
            # sinais
            rsi_period=int(rsi_period),
            entry_rsi_op="<=",
            entry_rsi_value=float(entry_rsi_value),
            exit_rsi_enabled=bool(exit_rsi_enabled),
            exit_rsi_op=">=",
            exit_rsi_value=float(exit_rsi_value),
            take_profit=float(take_profit) if take_profit > 0 else None,
            stop_loss=float(stop_loss) if stop_loss > 0 else None,
            # sizing
            order_size_type=order_size_type_internal,
            order_size_value=float(order_size_value_internal),
            pyramid=int(pyramid),
            # execução/custos
            order_type=order_type,
            limit_offset_ticks=int(limit_offset_ticks),
            tick_size=float(tick_size),
            commission_unit=commission_unit_internal,
            commission_value=float(commission_value),
            slippage_unit=slippage_unit,
            slippage_value=float(slippage_value),
        )

        # Simulação
        res = simulate_rule(df, float(bank), cfg)

        # ---------- Saídas ----------
        st.subheader("Métricas")
        st.json(res["stats"])

        st.subheader("Desempenho")
        perf_df = _performance_table(df, res)
        st.dataframe(perf_df, use_container_width=True)

        st.subheader("Equity Curve")
        eq = res["equity"]
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Preço + Sinais")
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preço"
            )
        )
        if len(res["signals"]) > 0:
            buys = res["signals"][res["signals"]["type"] == "buy"]
            sells = res["signals"][res["signals"]["type"] == "sell"]
            if len(buys) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buys["time"], y=buys["price"], mode="markers",
                        marker_symbol="triangle-up", marker_size=10, name="Buy"
                    )
                )
            if len(sells) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sells["time"], y=sells["price"], mode="markers",
                        marker_symbol="triangle-down", marker_size=10, name="Sell"
                    )
                )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(res["trades"])

        st.subheader("Resumo do Capital")
        st.write(
            f"Moeda: {base_currency} | Inicial: {res['initial_bank']:.2f} | "
            f"Final: {res['final_bank']:.2f} | Resultado: {res['final_bank'] - res['initial_bank']:.2f}"
        )
