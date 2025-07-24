import matplotlib.pyplot as plt
import numpy as np

def time_series_monitor(env, metrics, priority_map, interval, records):
    while True:
        now = env.now
        tps = metrics.throughputs(now)

        # Compute average throughput per tech, safely
        avg_tp = {}
        for tech in ['WiFi', 'NR-U']:
            values = [tp for name, tp in tps.items() if name.startswith(tech)]
            avg_tp[tech] = np.mean(values) if values else 0.0

        # Compute Jain's fairness index (now-safe)
        fairness = metrics.fairness(now)

        # Class-based fairness
        class_fair = metrics.fairness_by_priority(priority_map, now)

        # Store all time-series values
        records['time'].append(now)
        records['wifi_tp'].append(avg_tp['WiFi'])
        records['nru_tp'].append(avg_tp['NR-U'])
        records['jfi'].append(fairness)
        records['class_fair'].append(class_fair)

        yield env.timeout(interval)


          
def Single_scenario_Plot(records, cells_labeled, delay_values, delays, starved, m,colors ):
    plt.figure()
    plt.plot(records['time'], records['wifi_tp'], label='WiFi')
    plt.plot(records['time'], records['nru_tp'], label='NR-U')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Throughput (tx/s)")
    plt.title("Throughput Over Time")
    plt.legend()
    plt.grid(True)

    # Plot Jain's Fairness Index
    plt.figure()
    plt.plot(records['time'], records['jfi'], color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("Jain’s Fairness Index")
    plt.title("Fairness Over Time")
    plt.ylim(0, 1.05)
    plt.grid(True)
    primary = [cf['primary'] for cf in records['class_fair']]
    secondary = [cf['secondary'] for cf in records['class_fair']]
    plt.figure()
    plt.plot(records['time'], primary, label="Primary")
    plt.plot(records['time'], secondary, label="Secondary")
    plt.title("Fairness by Class")
    plt.xlabel("Time (s)")
    plt.ylabel("Jain's Index")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

   # --- Delay bar chart ---
    plot_delay_bar_chart(cells_labeled, delay_values, delays, starved, m)


    plt.show()
def Multiple_scenario_Plot(x,wifi_thr,nru_thr,
                           labels,cells_labeled,delays,
                           fairness_by_class_list,m,
                           share_labels,fairness,
                           delay_values,starved):
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

    plot_delay_bar_chart(cells_labeled, delay_values, delays, starved, m)
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



def get_tech_color(cell_name):
    if cell_name.startswith("WiFi"):
        return "blue"
    elif cell_name.startswith("NR-U"):
        return "orange"
    return "gray"

def plot_delay_bar_chart(cells_labeled, delay_values, delays, starved, m):
    # Detect original names (before labels) to color by tech
    cell_names = list(delays.keys())
    colors = [get_tech_color(name) for name in cell_names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(cells_labeled, delay_values, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Cell Name")
    plt.ylabel("Average Packet Delay (ms)")
    plt.title("User-Level Delay and Starvation")
    plt.grid(True)

    for bar, cell in zip(bars, cell_names):
        if cell in starved:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 'Starved',
                     ha='center', va='bottom', color='red', fontsize=8)

    for bar, cell_name in zip(bars, cell_names):
        t_val = m.get_t_value(cell_name)
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"T={t_val}", ha='center', va='bottom', fontsize=8, color='black')