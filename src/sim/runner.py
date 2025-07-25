import simpy
from .channel import Channel, BaseStation
from .cell import Cell
from .metrics import Metrics
from .Visulization import (
    time_series_monitor,
    Single_scenario_Plot,
    Multiple_scenario_Plot,
)
import numpy as np
import random


def simulate(wifi_count, nru_count, sim_time=2.0):
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

    wifi_bs = BaseStation(
        env,
        channel,
        "WiFi-BS",
        "WiFi",
        position=(0, -1),
        band="6GHz",
        ed_threshold_dBm=-62,
    )
    nru_bs = BaseStation(
        env,
        channel,
        "5G-BS",
        "NR-U",
        position=(0, 2),
        band="6GHz",
        ed_threshold_dBm=-72,
        cw_min=4,
        cw_max=32,
    )
    wifi_bs.metrics = metrics
    nru_bs.metrics = metrics
    # Instantiate and attach cells
    cells = []
    total = wifi_count + nru_count
    coords = [
        (random.uniform(0, 10*max(wifi_count, nru_count)),
         random.uniform(0, 10*max(wifi_count, nru_count)))
        for _ in range(total)
    ]
    random.shuffle(coords)
    for i in range(wifi_count):
        pos = coords.pop()
        pw = 2.0 if (i % 2 == 0) else 1.0
        cell = Cell(
            env,
            name=f"WiFi-{i+1}",
            tech="WiFi",
            channel=channel,
            position=pos,
            priority_weight=pw,
            phy_rate_bps=300e6,
        )
        wifi_bs.attach(cell)
        jitter = random.uniform(-2.0, +2.0)
        cell.ed_threshold = wifi_bs.ed_threshold + jitter
        cells.append(cell)
    for j in range(nru_count):
        pos = coords.pop()
        pw = 2.0 if j < nru_count // 2 else 1.0
        cell = Cell(
            env,
            name=f"NR-U-{j+1}",
            tech="NR-U",
            channel=channel,
            position=pos,
            priority_weight=pw,
            phy_rate_bps=600e6,
            cw_min=4 if pw >= 1.5 else 8,
            cw_max=32 if pw >= 1.5 else 64,
        )
        nru_bs.attach(cell)
        jitter = random.uniform(-2.0, +2.0)
        cell.ed_threshold = nru_bs.ed_threshold + jitter
        cells.append(cell)

    # 1. compute airtime shares
    wifi_weight = sum(c.priority_weight for c in cells if c.tech == "WiFi")
    nru_weight = sum(c.priority_weight for c in cells if c.tech == "NR-U")
    tot_w = wifi_weight + nru_weight
    wifi_bs.global_share = wifi_weight / tot_w
    nru_bs.global_share = nru_weight / tot_w

    # 2. pick your NAV/backoff‐grant parameters
    busy_th = 3  # number of consecutive busy‐samples before NAV
    # you could also make busy_th a parameter to simulate()

    # 3. compute each BS’s microsecond‐scale slot_time
    slot_time_wifi = 9e-6 / wifi_bs.global_share  # Wi-Fi slot = 9 μs
    slot_time_nru = 25e-6 / nru_bs.global_share  # NR-U LBT slot ≈25 μs

    # 4. start your monitors with those values
    env.process(
        wifi_bs.monitor(
            interval=slot_time_wifi, busy_threshold=busy_th, nav_duration=slot_time_wifi
        )
    )
    env.process(
        nru_bs.monitor(
            interval=slot_time_nru * 0.8,
            busy_threshold=busy_th - 1,
            nav_duration=slot_time_nru,
        )
    )

    # 5. fairness monitors (unchanged)
    env.process(wifi_bs.monitor_fairness(interval=slot_time_wifi))
    env.process(nru_bs.monitor_fairness(interval=slot_time_nru))

    def heartbeat():
        while True:
            print(f"Sim time: {env.now}")
            yield env.timeout(1.0)

    env.process(heartbeat())

    # Build grid and start cell processes
    grid = {cell.position: cell for cell in Cell.registry}
    for cell in cells:
        cell.grid = grid
        env.process(cell.run())
    priority_map = {cell.name: cell.priority_weight for cell in cells}
    records = {"time": [], "wifi_tp": [], "nru_tp": [], "jfi": [], "class_fair": []}

    # 🟢 Start the time-series monitor BEFORE running the simulation
    env.process(
        time_series_monitor(env, metrics, priority_map, interval=1.0, records=records)
    )

    # Execute simulation
    env.run(until=sim_time)
    metrics.stop(sim_time)

    return metrics, priority_map, cells, records


def main():
    # max_users = 10
    wifi_thr = []
    nru_thr = []
    fairness = []
    # user_counts = list(range(1, max_users+1))

    # print(f"Running parameter sweep for 1 to {max_users} users per tech...")
    scenarios = [
        (5, 5),
        # (10, 5),
        # (7, 15)
        # (8, 2),
        # (6, 4)
        # (20, 12),
        # (15, 25),
        # (30, 18)
    ]
    for wifi_n, nru_n in scenarios:
        m, priority_map, cells, records = simulate(wifi_n, nru_n)
        m.cell_map = {cell.name: cell for cell in cells}
        fairness_by_class_list = []
        share_labels = []
        rep = m.report()
        print("Per-cell WiFi Throughput:")
        for name, tp in rep["per_cell_throughput"].items():
            if name.startswith("WiFi"):
                print(f"  {name}: {tp:.1f} tx/s")
        starved = set(rep["starved_cells"])
        delays = rep["average_delay_per_cell"]
        # Add (P) or (S) label based on priority
        cells_labeled = [
            f"{cell} (P)" if priority_map.get(cell, 1.0) >= 1.5 else f"{cell} (S)"
            for cell in delays.keys()
        ]

        delay_values = [delays[cell] for cell in delays.keys()]
        colors = ["red" if cell in starved else "blue" for cell in delays.keys()]
        wifi_thr.append(rep["avg_throughput_per_tech"]["WiFi"])
        nru_thr.append(rep["avg_throughput_per_tech"]["NR-U"])
        fairness.append(rep["fairness"])
        class_fair = m.fairness_by_priority(priority_map)
        fairness_by_class_list.append(class_fair)
        share_labels.append(f"WiFi={wifi_n}, NR-U={nru_n}")
    #     print(f"Fairness (Primary): {class_fair['primary']:.2f}, "
    #           f"(Secondary): {class_fair['secondary']:.2f}")
    #     print(f"WiFi={wifi_n}, NR-U={nru_n} → "
    #           f"Thr WiFi={rep['avg_throughput_per_tech']['WiFi']:.2f}, "
    #           f"Thr NR-U={rep['avg_throughput_per_tech']['NR-U']:.2f}, "
    #           f"Fairness={rep['fairness']:.2f}")
    # wifi_users = [w for (w,_) in scenarios]
    # nru_users  = [n for (_,n) in scenarios]
    # Plot throughput vs. user count
    x = np.arange(len(scenarios))
    labels = [f"{w}/{n}" for w, n in scenarios]
    # print("\nFINAL T_dynamic values:")
    # for cell in cells:
    #     print(f"  {cell.name}: T_dynamic = {cell.T_dynamic}")
    if len(scenarios) == 1:
        Single_scenario_Plot(
            records, cells_labeled, delay_values, delays, starved, m, colors
        )
    else:
        Multiple_scenario_Plot(
            x,
            wifi_thr,
            nru_thr,
            labels,
            cells_labeled,
            delays,
            fairness_by_class_list,
            m,
            share_labels,
            fairness,
            delay_values,
            starved,
        )


if __name__ == "__main__":
    main()
