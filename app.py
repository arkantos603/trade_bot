import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data import load_ohlc_safe, add_returns
from src.sim_rule import StrategyConfig, simulate_rule

# ========= Helpers de formatação (pt-BR) =========
def _fmt_datetime_pt(ts) -> str:
    ts = pd.to_datetime(ts)
    meses = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]
    return f"{ts.day:02d} de {meses[ts.month-1]} de {ts.year}, {ts.hour:02d}:{ts.minute:02d}"

# ========= Tabela de Trades (formato tabela, sem HTML) =========
def _build_trades_df(df_price: pd.DataFrame, res: dict, base_currency="USD") -> pd.DataFrame:
    """
    Retorna um DataFrame com duas linhas por trade (Entrada e Saída),
    colunas em USD e %, e **banca total** após cada trade (últimas colunas).
    """
    trades = res.get("trades", pd.DataFrame()).copy()
    if trades is None or len(trades) == 0:
        return pd.DataFrame()

    trades = trades.sort_values("exit_time").reset_index(drop=True)
    initial = float(res.get("initial_bank", 1.0)) or 1.0

    rows = []
    cum = 0.0  # L&P acumulado (USD)

    for i, r in trades.iterrows():
        trade_no = i + 1
        e_time, x_time = r["entry_time"], r["exit_time"]
        e_price, x_price = float(r["entry_price"]), float(r["exit_price"])
        qty = float(r["qty"])

        base_notional = e_price * qty if (e_price > 0 and qty > 0) else 0.0
        pnl_cash = float(r.get("pnl_cash", (x_price - e_price) * qty))
        pnl_pct = (pnl_cash / base_notional) if base_notional > 0 else 0.0

        # MFE/MAE entre entrada e saída
        try:
            win = df_price.loc[e_time:x_time]
        except Exception:
            win = df_price[(df_price.index >= e_time) & (df_price.index <= x_time)]
        if len(win) == 0:
            max_high, min_low = e_price, e_price
        else:
            max_high = float(win["High"].max())
            min_low  = float(win["Low"].min())
        runup_pct  = (max_high - e_price) / e_price if e_price > 0 else 0.0
        dd_pct     = (min_low  - e_price) / e_price if e_price > 0 else 0.0
        runup_cash = runup_pct * base_notional
        dd_cash    = dd_pct * base_notional

        cum_after = cum + pnl_cash
        bank_before = initial + cum           # banca antes de fechar o trade (linha Entrada)
        bank_after  = initial + cum_after     # banca depois de fechar o trade (linha Saída)

        reason = str(r.get("reason", "") or "").upper()
        saida_label = "Saída" if reason == "" else f"Saída ({reason})"

        # Linha ENTRADA
        rows.append({
            "Trade #": trade_no,
            "Viés": "Viés de alta",
            "Tipo": "Entrada",
            "Data/Tempo": _fmt_datetime_pt(e_time),
            "Sinal": "Compra",
            "Preço (USD)": e_price,
            "Qty": qty,
            "Notional (USD)": base_notional,
            "P&L (USD)": None,
            "P&L (%)": None,
            "Run-up (USD)": None,
            "Run-up (%)": None,
            "Drawdown (USD)": None,
            "Drawdown (%)": None,
            "L&P acumulado (USD)": cum,
            "L&P acumulado (%)": (cum / initial) if initial > 0 else 0.0,
            "Banca total (USD)": bank_before,
            "Banca total (%)": (bank_before / initial - 1.0) if initial > 0 else 0.0,
        })

        # Linha SAÍDA
        rows.append({
            "Trade #": trade_no,
            "Viés": "Viés de alta",
            "Tipo": saida_label,
            "Data/Tempo": _fmt_datetime_pt(x_time),
            "Sinal": "Fechar posição",
            "Preço (USD)": x_price,
            "Qty": qty,
            "Notional (USD)": base_notional,
            "P&L (USD)": pnl_cash,
            "P&L (%)": pnl_pct,
            "Run-up (USD)": runup_cash,
            "Run-up (%)": runup_pct,
            "Drawdown (USD)": dd_cash,
            "Drawdown (%)": dd_pct,
            "L&P acumulado (USD)": cum_after,
            "L&P acumulado (%)": (cum_after / initial) if initial > 0 else 0.0,
            "Banca total (USD)": bank_after,
            "Banca total (%)": (bank_after / initial - 1.0) if initial > 0 else 0.0,
        })

        cum = cum_after

    # Ordena por trade desc e garante ordem das colunas
    out = pd.DataFrame(rows).sort_values(["Trade #", "Tipo"], ascending=[False, True]).reset_index(drop=True)

    # Converte % para base 0–100
    pct_cols = ["P&L (%)", "Run-up (%)", "Drawdown (%)", "L&P acumulado (%)", "Banca total (%)"]
    for c in pct_cols:
        out[c] = out[c].astype(float) * 100.0

    col_order = [
        "Trade #","Viés","Tipo","Data/Tempo","Sinal","Preço (USD)","Qty","Notional (USD)",
        "P&L (USD)","P&L (%)","Run-up (USD)","Run-up (%)","Drawdown (USD)","Drawdown (%)",
        "L&P acumulado (USD)","L&P acumulado (%)",
        "Banca total (USD)","Banca total (%)"   # << últimas colunas
    ]
    return out[col_order]

# ========= Tabela de desempenho =========
def _fmt_usd_pct(val_cash: float, pct_base: float) -> str:
    if pct_base == 0:
        pct = 0.0
    else:
        pct = 100.0 * (val_cash / pct_base)
    sign = "+" if val_cash > 0 else ""
    return f"{val_cash:,.2f} USD ({sign}{pct:.2f}%)"

def _performance_table(df_price: pd.DataFrame, res: dict) -> pd.DataFrame:
    eq = res["equity"]
    perf = res.get("perf", {})
    initial = float(res["initial_bank"])
    final = float(res["final_bank"])

    net_profit_cash = final - initial
    open_pnl_cash = float(perf.get("open_pnl_cash", 0.0))
    gross_profit = float(perf.get("gross_profit", 0.0))
    gross_loss = float(perf.get("gross_loss", 0.0))
    fees_paid = float(perf.get("fees_paid", 0.0))
    max_qty_held = float(perf.get("max_qty_held", 0.0))

    p0 = float(df_price["Close"].iloc[0])
    p1 = float(df_price["Close"].iloc[-1])
    bh_final = initial * (p1 / p0) if p0 != 0 else initial
    bh_profit = bh_final - initial

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

# -------- Cache (30 min) --------
@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_prices(ticker: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
    df = load_ohlc_safe(ticker, start=start, end=end, interval=interval)
    return add_returns(df)

# =================== UI ===================
st.set_page_config(page_title="PPO Trading + Rule Simulator (PnL-first)", layout="wide")
st.title("PPO Trading + Simulador por Regras — Foco em Lucratividade (Dados: Binance)")

with st.sidebar:
    st.header("Dados")
    ticker = st.text_input("Par (ex.: BTC-USD, ETH-USD, SOL-USD)", "BTC-USD")
    start = st.date_input("Início", pd.to_datetime("2018-01-01"))
    end = st.date_input("Fim", pd.to_datetime("2024-12-31"))

    st.markdown("---")
    st.subheader("Tempos Gráficos")
    tf_options = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]
    strategy_interval = st.selectbox("Intervalo da estratégia (backtest)", tf_options, index=tf_options.index("1d"))
    chart_interval = st.selectbox(
        "Intervalo do gráfico (visualização)", tf_options, index=tf_options.index("1d"),
        help="Não altera a simulação; só muda a resolução do candleplot."
    )

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
        help="Ex.: BTCUSDT 0.1/0.01 conforme exchange/par.",
    )

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

# =================== Execução ===================
if st.button("Rodar Simulação"):
    with st.spinner("Baixando candles (Binance) e simulando..."):
        # Dados para ESTRATÉGIA
        try:
            df = _fetch_prices(ticker, str(start), str(end), strategy_interval)
        except Exception as e:
            msg = str(e)
            if ("429" in msg) or ("Too Many Requests" in msg) or ("-1003" in msg):
                st.warning(
                    "A API pública da Binance limitou as requisições desta máquina (HTTP 429). "
                    "Aguarde ~1–2 minutos e rode novamente. O app usa cache por 30 min para evitar excesso."
                )
            else:
                st.error(
                    "Não consegui baixar dados para esse par/período no intervalo selecionado.\n\n"
                    "Exemplos de pares aceitos (digite com '-USD', eu converto para USDT):\n"
                    "- BTC-USD → BTCUSDT\n- ETH-USD → ETHUSDT\n- SOL-USD → SOLUSDT\n\n"
                    f"Detalhes: {e}"
                )
            st.stop()

        # mapeia UI -> config interna
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

        # Simulação no intervalo da estratégia
        res = simulate_rule(df, float(bank), cfg)

        # ----- Saídas -----
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

        # Dados para GRÁFICO (visualização), sem afetar simulação
        try:
            df_vis = _fetch_prices(ticker, str(start), str(end), chart_interval)
        except Exception:
            df_vis = df  # fallback

        st.subheader(f"Preço + Sinais (visualização: {chart_interval}, estratégia: {strategy_interval})")
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df_vis.index, open=df_vis["Open"], high=df_vis["High"],
                low=df_vis["Low"], close=df_vis["Close"], name="Preço"
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

        # -------- Trades (detalhado) em TABELA --------
        st.subheader("Trades (detalhado)")
        trades_view = _build_trades_df(df, res, base_currency)
        if len(trades_view) == 0:
            st.info("Sem trades no período.")
        else:
            # Configura formatação numérica (se sua versão do Streamlit suportar column_config)
            try:
                st.dataframe(
                    trades_view,
                    use_container_width=True,
                    column_config={
                        "Preço (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "Qty": st.column_config.NumberColumn(format="%.6f"),
                        "Notional (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "P&L (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Run-up (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "Run-up (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Drawdown (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "L&P acumulado (USD)": st.column_config.NumberColumn(format="%.2f"),
                        "L&P acumulado (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Banca total (USD)": st.column_config.NumberColumn(format="%.2f"),

                    },
                )
            except Exception:
                # fallback: sem column_config (versões antigas do streamlit)
                st.dataframe(trades_view, use_container_width=True)

        st.subheader("Resumo do Capital")
        st.write(
            f"Moeda: {base_currency} | Inicial: {res['initial_bank']:.2f} | "
            f"Final: {res['final_bank']:.2f} | Resultado: {res['final_bank'] - res['initial_bank']:.2f}"
        )
