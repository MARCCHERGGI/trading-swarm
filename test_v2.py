from trading_swarm import create_swarm, LiveScanner

swarm = create_swarm()
print(f"Agents ({len(swarm.agents)}): {[a.name for a in swarm.agents]}")

scanner = LiveScanner(swarm)
sigs = scanner.scan(['AAPL', 'BTC-USD', 'NVDA', 'SPY'])
print("\nLive signals with sentiment:")
for s in sigs:
    print(f"  {s['ticker']}: {s['action']} {float(s['consensus']):.3f}")
    # Show per-agent breakdown
    for agent_name, vote in zip([a.name for a in swarm.agents], [s.get('sma',0), s.get('rsi',0), s.get('momentum',0), s.get('bollinger',0), s.get('volume',0), s.get('macd',0)]):
        print(f"    {agent_name}: {vote}")
