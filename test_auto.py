from trading_swarm import create_swarm, LiveScanner
from alpaca_trader import AlpacaTrader

swarm = create_swarm()
scanner = LiveScanner(swarm)
signals = scanner.scan(['AAPL', 'BTC-USD', 'SPY', 'MSFT', 'TSLA'])

print('Signals summary:')
for s in signals:
    print(f"  {s['ticker']}: consensus={s['consensus']:.3f} action={s['action']}")

trader = AlpacaTrader()
print(f"\nMode: {trader.mode}")
trader.auto_trade(signals, portfolio_pct=0.05)
positions = trader.get_positions()
print(f"Positions after auto_trade: {len(positions)}")
for p in positions:
    print(f"  {p['ticker']}: {p['side']} {p['qty']} shares @ ${p['entry_price']:.2f}")
pnl = trader.get_pnl()
print(f"\nEquity: ${pnl['equity']:,.2f}  Cash: ${pnl['cash']:,.2f}  PnL: {pnl['total_return_pct']:.2f}%")
