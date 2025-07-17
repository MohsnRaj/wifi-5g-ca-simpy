import matplotlib.pyplot as plt

def plot_spectrum_access(events, title="Spectrum Access Timeline", time_unit="s"):
    """
    Plot a timeline of spectrum access events for Wi-Fi and NR-U users.

    Args:
        events: List of tuples (tech, start_time, duration) where:
            - tech: str, either 'WiFi' or 'NR-U'
            - start_time: float, time when transmission starts
            - duration: float, length of the transmission
        title: Title of the plot
        time_unit: Label for the time axis (default 's' for seconds)

    Usage:
        events = [
            ('WiFi', 0.1, 0.01),
            ('NR-U', 0.12, 0.005),
            ...
        ]
        plot_spectrum_access(events)
    """
    # Assign vertical positions for each technology
    levels = {'WiFi': 10, 'NR-U': 20}
    height = 8  # height of the bar for each event

    fig, ax = plt.subplots(figsize=(10, 3))

    # Keep track of legend entries
    seen = set()
    for tech, start, duration in events:
        level = levels.get(tech, 0)
        label = tech if tech not in seen else None
        ax.broken_barh([(start, duration)], (level - height/2, height),
                       facecolors='tab:blue' if tech=='WiFi' else 'tab:orange',
                       label=label)
        seen.add(tech)

    ax.set_yticks(list(levels.values()))
    ax.set_yticklabels(list(levels.keys()))
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
