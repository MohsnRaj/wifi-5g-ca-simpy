import math
from collections import defaultdict


def shot_noise(x):
    # placeholder for noise modeling
    return 1e-9


class Metrics:
    """
    Collects per-cell transmission events, success/loss statistics,
    computes per-cell and per-tech throughput, and Jain's fairness.
    """

    def __init__(self):
        # per-cell lists of transmission timestamps
        self.tx_times = defaultdict(list)
        # per-cell counts of successful and lost packets
        self.packet_success = defaultdict(int)
        self.packet_loss = defaultdict(int)
        self.start_time = None
        self.stop_time = None
        self.delay_records = defaultdict(list)
        self.last_success_time = {}
        self.starvation_threshold = 2.0  # seconds

    def get_t_value(self, cell_name):
        if hasattr(self, "cell_map") and cell_name in self.cell_map:
            return getattr(self.cell_map[cell_name], "T_dynamic", "?")
        return "?"

    def record_delay(self, cell_name: str, delay: float):
        self.delay_records[cell_name].append(delay)

    def start(self, t0: float):
        """Mark simulation start time"""
        self.start_time = t0

    def record_tx(self, cell_name: str, t: float):
        """Log that cell_name transmitted at time t"""
        self.tx_times[cell_name].append(t)

    def record_success(self, cell_name: str):
        self.packet_success[cell_name] += 1
        self.last_success_time[cell_name] = self.stop_time or self.start_time

    def record_loss(self, cell_name: str):
        """Increment lost packet count for cell_name"""
        self.packet_loss[cell_name] += 1

    def stop(self, t1: float):
        """Mark simulation end time"""
        self.stop_time = t1

    def throughputs(self, now) -> dict:
        """
    Compute per-cell *instantaneous* throughput as transmitted packets / duration
    since the last call, for proper time-series plotting.
    """
        # اولین بار که فراخوانی می‌شود، مقادیر اولیه را ست کن
        if not hasattr(self, '_last_tp_time'):
            self._last_tp_time   = self.start_time
            self._last_tx_counts = {cell: 0 for cell in self.tx_times}

        # طول بازه زمانی
        delta = max(1e-6, now - self._last_tp_time)

        tps = {}
        for cell, times in self.tx_times.items():
            prev_count = self._last_tx_counts.get(cell, 0)
            sent = len(times) - prev_count
            tps[cell] = sent / delta

    # به‌روزرسانی state برای نوبت بعدی
        self._last_tp_time   = now
        self._last_tx_counts = {cell: len(times) for cell, times in self.tx_times.items()}

        return tps
    def cumulative_throughputs(self, now=None) -> dict:
        """
        Compute average throughput over entire simulation (from start to now or stop).
        Used for final reports.
        """
        end_time = self.stop_time if self.stop_time is not None else now
        duration = max(1e-6, end_time - self.start_time)
        return {cell: len(times) / duration for cell, times in self.tx_times.items()}
    def fairness(self, now=None) -> float:
        """Jain's index over per-cell instantaneous throughput"""
        tp = list(self.throughputs(now).values()) 
        if not tp:
            return 0.0
        s1 = sum(tp)
        s2 = sum(x * x for x in tp)
        N = len(tp)
        return (s1 * s1) / (N * s2) if s2 > 0 else 0.0

    def final_fairness(self) -> float:
        """Jain's index over total cumulative throughput (برای گزارش نهایی)"""
        tp = list(self.cumulative_throughputs().values())
        if not tp:
            return 0.0
        s1 = sum(tp)
        s2 = sum(x * x for x in tp)
        N = len(tp)
        return (s1 * s1) / (N * s2) if s2 > 0 else 0.0

    def report(self) -> dict:
        """
        Generate report dict with:
         - per_cell_throughput
         - avg_throughput_per_tech
         - fairness
         - total_cells
         - packet_loss_rate_per_cell
        """
        # Compute per-cell throughput
        tps = self.cumulative_throughputs()

        # Compute average throughput per tech
        tech_tot = defaultdict(list)
        for cell, tp in tps.items():
            # use rsplit to support tech names with hyphens (e.g., 'NR-U')
            tech = cell.rsplit("-", 1)[0]
            tech_tot[tech].append(tp)
        avg = {
            tech: (sum(vals) / len(vals) if vals else 0.0)
            for tech, vals in tech_tot.items()
        }
        avg_delay = {
            cell: (sum(d) / len(d)) * 1000 if d else 0.0  # ← تبدیل به ms
            for cell, d in self.delay_records.items()
        }

        # Starvation detection
        starved = []
        for cell in self.packet_success:
            last_time = self.last_success_time.get(cell, self.start_time)
            if self.stop_time - last_time >= self.starvation_threshold:
                starved.append(cell)

        # Compute packet loss rates per cell
        loss_rate = {}
        cells = set(self.packet_success) | set(self.packet_loss) | set(tps)
        for cell in cells:
            total_pkts = self.packet_success.get(cell, 0) + self.packet_loss.get(
                cell, 0
            )
            loss_rate[cell] = (
                (self.packet_loss.get(cell, 0) / total_pkts) if total_pkts > 0 else 0.0
            )

        return {
            "per_cell_throughput": tps,
            "avg_throughput_per_tech": avg,
            "average_delay_per_cell": avg_delay,
            "starved_cells": starved,
            "fairness": self.final_fairness(),
            "total_cells": len(tps),
            "packet_loss_rate_per_cell": loss_rate,
        }

    def fairness_by_priority(self, priority_map: dict, now=None) -> dict:
        """
        Compute Jain's fairness separately for primary and secondary users.
        priority_map: dict of cell_name -> priority_weight
        Returns: dict with 'primary' and 'secondary' fairness indices
        """
        # Separate throughputs
        tps = self.cumulative_throughputs(now)
        primary = []
        secondary = []

        for cell, tp in tps.items():
            if priority_map.get(cell, 1.0) >= 1.5:
                primary.append(tp)
            else:
                secondary.append(tp)

        def jain(tp_list):
            if not tp_list:
                return 0.0
            s1 = sum(tp_list)
            s2 = sum(x * x for x in tp_list)
            N = len(tp_list)
            return (s1 * s1) / (N * s2) if s2 > 0 else 0.0

        return {"primary": jain(primary), "secondary": jain(secondary)}
