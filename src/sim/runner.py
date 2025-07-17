import simpy
from .channel import Channel
from .cell import Cell


def main():
    env = simpy.Environment()
    channel = Channel(env)

    # operator configuration
    wifi_count = 3
    nru_count = 2
    cells = []
    for i in range(1, wifi_count+1):
        cells.append(Cell(env, f"WiFi-{i}", "WiFi", channel))
    for j in range(1, nru_count+1):
        cells.append(Cell(env, f"5G-{j}", "NR-U", channel))

    SIM_TIME = 20
    print(f"Starting simulation for {SIM_TIME} time units...\n")
    env.run(until=SIM_TIME)
    print("\nSimulation complete. Summary:")
    for cell in cells:
        print(f"{cell.name} ({cell.tech}): TXs={cell.tx_count}, backoffs={cell.backoff_count}, queue={cell.state['queue_len']}")

if __name__ == "__main__":
    main()
