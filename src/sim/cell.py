import simpy
import random
import numpy as np

class Cell:
    """
    A network cell node in the SimPy-based CA simulator.
    Each cell registers itself and discovers neighbors.
    It tracks its own state, exchanges metrics,
    and applies a CA update rule before each slot.
    """
    registry = []  # registry of all cells

    def __init__(self, env: simpy.Environment, name: str, tech: str, channel, model=None):
        self.env = env
        self.name = name
        self.tech = tech  # "WiFi" or "NR-U"
        self.channel = channel
        self.model = model

        Cell.registry.append(self)
        self.neighbors = []

        # State and summary counters
        self.state = {
            'queue_len': 0,
            'last_tx': False,
            'cw': 4 if tech == 'NR-U' else 16,
        }
        self.tx_count = 0
        self.backoff_count = 0

        env.process(self.run())

    def discover_neighbors(self):
        self.neighbors = [c for c in Cell.registry if c is not self]

    def update(self):
        # On first call, discover neighbors
        if not self.neighbors:
            self.discover_neighbors()
        # Simple fairness heuristic
        avg_nbr_queue = sum(n.state['queue_len'] for n in self.neighbors) / len(self.neighbors) if self.neighbors else 0
        if self.state['queue_len'] > avg_nbr_queue:
            self.state['cw'] = max(1, self.state['cw'] // 2)
        else:
            self.state['cw'] = min(64, self.state['cw'] * 2)

    def generate_traffic(self):
        """
        Poisson arrivals per time unit (lam=0.5).
        """
        return np.random.poisson(lam=0.5)

    def run(self):
        slot_time = 0.001 if self.tech == 'NR-U' else 0.009
        while True:
            # CA update
            self.update()

            # Backoff countdown before sensing
            backoff_slots = random.randint(1, self.state['cw'])
            backoff_time = backoff_slots * slot_time
            self.backoff_count += 1
            yield self.env.timeout(backoff_time)

            # Sense channel
            if self.channel.is_idle():
                yield self.channel.occupy()
                self.state['last_tx'] = True
                self.tx_count += 1
                print(f"{self.env.now:.3f}: {self.name} ({self.tech}) TX #{self.tx_count} CW={self.state['cw']}")
                yield self.env.timeout(1)
                yield self.channel.release()
            else:
                self.state['last_tx'] = False

            # Traffic arrivals at end of slot
            arrivals = self.generate_traffic()
            self.state['queue_len'] += arrivals