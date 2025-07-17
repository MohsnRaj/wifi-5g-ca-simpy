import simpy
import matplotlib.pyplot as plt
from .channel import Channel, BaseStation
from .cell import Cell
from .metrics import Metrics
import numpy as np


def simulate(wifi_count, nru_count, sim_time=20.0, T=4, delta=1):
    """
    Run a single simulation scenario with given number of Wi-Fi and NR-U users.
    Returns a Metrics instance with collected results.
    """
    
    # Initialize metrics and environment
    metrics = Metrics()
    metrics.start(0.0)
    env = simpy.Environment()
    channel = Channel(env)

    # Wrap occupy to record TX events
    original_occupy = channel.occupy
    def record_occupy(band, tx_cell, duration):
        start = env.now
        original_occupy(band, tx_cell, duration)
        # track per-cell
        metrics.record_tx(tx_cell.name, start)
    channel.occupy = record_occupy

    # Create base stations
    
    wifi_bs = BaseStation(env, channel, "WiFi-BS", "WiFi", position=(0,-1), band="6GHz", ed_threshold_dBm=-62)
    nru_bs  = BaseStation(env, channel, "5G-BS",  "NR-U",  position=(0, 2),  band="6GHz", ed_threshold_dBm=-72)
    env.process(wifi_bs.monitor(interval=0.1))
    env.process(nru_bs.monitor(interval=0.1))
    # Instantiate and attach cells
    cells = []
    # مثال: نصف وای‌فای‌ها streaming (high) و نصف browsing (low)
    for i in range(wifi_count):
        pos = (i, 0)
        pw = 2.0 if (i % 2 == 0) else 1.0
        cell = Cell(env,
                    name=f"WiFi-{i+1}",
                    tech="WiFi",
                    channel=channel,
                    position=pos,
                    priority_weight=pw)
        wifi_bs.attach(cell)
        cells.append(cell)
    for j in range(nru_count):
        pos = (j, 1)
        pw = 2.0 if j < nru_count//2 else 1.0
        cell = Cell(env,
                    name=f"NR-U-{j+1}",
                    tech="NR-U",
                    channel=channel,
                    position=pos,
                    priority_weight=pw)
        nru_bs.attach(cell)
        cells.append(cell)

    total = wifi_bs.user_count + nru_bs.user_count
    wifi_bs.global_share = wifi_bs.user_count / total   # سهم WiFi
    nru_bs.global_share  = nru_bs.user_count  / total   # سهم NR-U
    print(f"Shares → WiFi: {wifi_bs.global_share:.2f}, NR-U: {nru_bs.global_share:.2f}")

    # Build grid and start cell processes
    grid = {cell.position: cell for cell in Cell.registry}
    for cell in cells:
        cell.grid = grid
        env.process(cell.run())

    # Execute simulation
    env.run(until=sim_time)
    metrics.stop(sim_time)
    return metrics


def main():
    max_users = 10
    wifi_thr = []
    nru_thr = []
    fairness = []
    # user_counts = list(range(1, max_users+1))

    print(f"Running parameter sweep for 1 to {max_users} users per tech...")
    scenarios = [
        (2, 8),
        (4, 6)
        # (5, 5),
        # (6, 4),
        # (8, 2),
    ]
    for wifi_n, nru_n in scenarios:
        m = simulate(wifi_n, nru_n)
        rep = m.report()
        wifi_thr.append(rep['avg_throughput_per_tech']['WiFi'])
        nru_thr.append(rep['avg_throughput_per_tech']['NR-U'])
        fairness.append(rep['fairness'])
        print(f"WiFi={wifi_n}, NR-U={nru_n} → "
              f"Thr WiFi={rep['avg_throughput_per_tech']['WiFi']:.2f}, "
              f"Thr NR-U={rep['avg_throughput_per_tech']['NR-U']:.2f}, "
              f"Fairness={rep['fairness']:.2f}")
    wifi_users = [w for (w,_) in scenarios]
    nru_users  = [n for (_,n) in scenarios]
    # Plot throughput vs. user count
    x      = np.arange(len(scenarios))
    labels = [f"{w}/{n}" for w,n in scenarios]  # e.g. "2/8", "4/6"

    # — Throughput plot —
    plt.figure()
    plt.plot(x, wifi_thr, 'o-', label='WiFi')
    plt.plot(x, nru_thr,  'o-', label='NR-U')
    plt.xticks(x, labels)                       # apply our labels
    plt.xlabel("Users (WiFi / NR-U)")
    plt.ylabel("Throughput (tx /s)")
    plt.title("Throughput vs. Load (6 GHz)")
    plt.legend()
    plt.grid(True)

    # — Fairness plot —
    plt.figure()
    plt.plot(x, fairness, 's-', color='purple')
    plt.xticks(x, labels)
    plt.xlabel("Users (WiFi / NR-U)")
    plt.ylabel("Jain’s Fairness Index")
    plt.title("Fairness vs. Load (6 GHz)")
    plt.ylim(0,1.05)
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()
