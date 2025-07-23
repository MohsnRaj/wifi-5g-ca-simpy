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
    nru_bs  = BaseStation(env, channel, "5G-BS",  "NR-U",  position=(0, 2),  band="6GHz", ed_threshold_dBm=-72, cw_min=4, cw_max=1024)
    wifi_bs.metrics = metrics
    nru_bs.metrics  = metrics
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
    priority_map = {cell.name: cell.priority_weight for cell in cells}
    # Execute simulation
    env.run(until=sim_time)
    metrics.stop(sim_time)
    return metrics, priority_map, cells


def main():
    max_users = 10
    wifi_thr = []
    nru_thr = []
    fairness = []
    # user_counts = list(range(1, max_users+1))

    print(f"Running parameter sweep for 1 to {max_users} users per tech...")
    scenarios = [
    (4, 6), 
    (5, 5), 
    (6, 4)
]
    for wifi_n, nru_n in scenarios:
        m, priority_map, cells = simulate(wifi_n, nru_n)
        m.cell_map = {cell.name: cell for cell in cells}   
        fairness_by_class_list = []
        share_labels = []
        rep = m.report()
        starved = set(rep['starved_cells'])
        delays = rep['average_delay_per_cell']
        # Add (P) or (S) label based on priority
        cells_labeled = [
            f"{cell} (P)" if priority_map.get(cell, 1.0) >= 1.5 else f"{cell} (S)"
            for cell in delays.keys()
        ]

        delay_values = [delays[cell] for cell in delays.keys()]
        colors = ['red' if cell in starved else 'blue' for cell in delays.keys()]
        wifi_thr.append(rep['avg_throughput_per_tech']['WiFi'])
        nru_thr.append(rep['avg_throughput_per_tech']['NR-U'])
        fairness.append(rep['fairness'])
        class_fair = m.fairness_by_priority(priority_map)
        fairness_by_class_list.append(class_fair)
        share_labels.append(f"WiFi={wifi_n}, NR-U={nru_n}")
        print(f"Fairness (Primary): {class_fair['primary']:.2f}, "
              f"(Secondary): {class_fair['secondary']:.2f}")
        print(f"WiFi={wifi_n}, NR-U={nru_n} → "
              f"Thr WiFi={rep['avg_throughput_per_tech']['WiFi']:.2f}, "
              f"Thr NR-U={rep['avg_throughput_per_tech']['NR-U']:.2f}, "
              f"Fairness={rep['fairness']:.2f}")
    wifi_users = [w for (w,_) in scenarios]
    nru_users  = [n for (_,n) in scenarios]
    # Plot throughput vs. user count
    x      = np.arange(len(scenarios))
    labels = [f"{w}/{n}" for w,n in scenarios]  
    print("\nFINAL T_dynamic values:")
    for cell in cells:
        print(f"  {cell.name}: T_dynamic = {cell.T_dynamic}")
    plot_fairness_by_priority(fairness_by_class_list, share_labels)

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

    plt.figure(figsize=(10, 5))
    bars = plt.bar(cells_labeled, delay_values, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Cell Name")
    plt.ylabel("Average Packet Delay (s)")
    plt.title("User-Level Delay and Starvation")
    plt.grid(True)

    # Annotate starved users
    for bar, cell in zip(bars, delays.keys()):
        if cell in starved:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 'Starved',
                    ha='center', va='bottom', color='red', fontsize=8)
    for bar, cell_name in zip(bars, delays.keys()):
        t_val = m.get_t_value(cell_name)
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"T={t_val}", ha='center', va='bottom', fontsize=8, color='black')
    plt.show()
def plot_fairness_by_priority(fairness_by_class_list, share_labels):
    primary_vals = [x['primary'] for x in fairness_by_class_list]
    secondary_vals = [x['secondary'] for x in fairness_by_class_list]

    x = range(len(share_labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], primary_vals, width, label='Primary', alpha=0.8)
    ax.bar([i + width/2 for i in x], secondary_vals, width, label='Secondary', alpha=0.8)

    ax.set_ylabel("Fairness (Jain's Index)")
    ax.set_xlabel("WiFi/NR-U Share")
    ax.set_title("Fairness by Priority Class")
    ax.set_xticks(x)
    ax.set_xticklabels(share_labels, rotation=30)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fairness_priority.png")
    plt.show()
if __name__ == '__main__':
    main()
