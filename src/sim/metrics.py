# metrics.py
import math
from collections import defaultdict

class Metrics:
    """
    Collects per-cell transmission events, computes per-cell throughput
    and Jain's fairness over all cells.
    """
    def __init__(self):
        self.tx_times = defaultdict(list)  # name -> list of tx timestamps
        self.start_time = None
        self.stop_time = None

    def start(self, t0: float):
        self.start_time = t0

    def record_tx(self, cell_name: str, t: float):
        """
        Log that cell_name transmitted at time t.
        """
        self.tx_times[cell_name].append(t)

    def stop(self, t1: float):
        self.stop_time = t1

    def throughputs(self):
        """
        Returns dict cell_name -> throughput (tx count / duration).
        """
        duration = self.stop_time - self.start_time
        return {
            name: len(times) / duration
            for name, times in self.tx_times.items()
        }

    def fairness(self):
        """
        Jain's fairness over all cells:
           J = (sum x_i)^2 / (N * sum x_i^2)
        where x_i is throughput of cell i.
        """
        tp = list(self.throughputs().values())
        if not tp:
            return 0.0
        N = len(tp)
        s1 = sum(tp)
        s2 = sum(x*x for x in tp)
        return (s1*s1) / (N * s2) if s2>0 else 0.0

    def report(self):
        tps = self.throughputs()
        avg = {}
        for tech in ('WiFi','NR-U'):
            vals = [v for name,v in tps.items() if name.startswith(tech)]
            avg[tech] = (sum(vals)/len(vals)) if vals else 0.0

        return {
            'per_cell_throughput': tps,
            'avg_throughput_per_tech': avg,
            'fairness': self.fairness(),
            'total_cells': len(tps)
        }
