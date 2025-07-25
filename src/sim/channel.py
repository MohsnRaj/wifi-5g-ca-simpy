import simpy
import math

class Channel:
    """
    Multi-band wireless channel with simple free-space propagation and collision model.

    Attributes:
        env: SimPy environment
        tx_power_dBm: Transmit power (dBm)
        pathloss_exponent: Path-loss exponent (unitless)
        noise_floor_dBm: Ambient noise floor (dBm)
        rx_sensitivity_dBm: Receiver sensitivity threshold (dBm)
        bands: tuple of frequency band identifiers (e.g., '2.4GHz', '5GHz', '6GHz')
    """
    def __init__(self,
                 env: simpy.Environment,
                 bands=('2.4GHz', '5GHz', '6GHz'),
                 tx_power_dBm=20,
                 pathloss_exponent=3.0,
                 noise_floor_dBm=-95,
                 rx_sensitivity_dBm=-82):
        self.env = env
        self.bands = bands
        self.max_capacity = 1.0  # Max 1 second of usage per 1 second simulation time
        self.used_capacity = 0.0
        self.tx_power = tx_power_dBm
        self.pl_e = pathloss_exponent
        self.noise_floor = noise_floor_dBm
        self.rx_sens = rx_sensitivity_dBm
        # ongoing transmissions: band -> list of (transmitter, end_time)
        self.transmissions = {band: [] for band in bands}

    def _cleanup(self, band):
        now = self.env.now
        self.transmissions[band] = [(tx, end) for (tx, end) in self.transmissions[band] if end > now]

    def is_idle(self, band: str, rx_cell, ed_threshold_dBm: float) -> bool:
        self._cleanup(band)
        for (tx, end) in self.transmissions[band]:
            pr = self.recv_power_dBm(tx, rx_cell, band)
            if pr >= ed_threshold_dBm:
                return False
        return True

    def occupy(self, band: str, tx_cell, duration: float):
        # Drop or defer if over channel capacity
        if self.used_capacity + duration > self.max_capacity:
            # Simulate congestion: Drop packet or apply delay penalty
            # Here: We'll drop the packet silently
            if hasattr(tx_cell, "metrics"):
                tx_cell.metrics.record_drop(tx_cell.name)  # optional
            return

        # Accept the transmission
        self.used_capacity += duration
        end_time = self.env.now + duration
        self.transmissions[band].append((tx_cell, end_time))

    def release(self, band: str, tx_cell):
        self.transmissions[band] = [(tx, end) for (tx, end) in self.transmissions[band] if tx is not tx_cell]

    def path_loss(self, cell_a, cell_b, band: str) -> float:
        dx = cell_a.position[0] - cell_b.position[0]
        dy = cell_a.position[1] - cell_b.position[1]
        d = max(1.0, math.hypot(dx, dy) * 10)  # grid spacing = 10m
        f_MHz = float(band.replace('GHz','')) * 1000
        fspl = 20 * math.log10(d) + 20 * math.log10(f_MHz) + 32.44
        return fspl + 10 * (self.pl_e - 2) * math.log10(d)

    def recv_power_dBm(self, tx_cell, rx_cell, band: str) -> float:
        pl = self.path_loss(tx_cell, rx_cell, band)
        return self.tx_power - pl

    def can_receive(self, band: str, tx_cell, rx_cell) -> bool:
        sig = 10 ** (self.recv_power_dBm(tx_cell, rx_cell, band) / 10)
        interf = 0.0
        for (other, end) in self.transmissions[band]:
            if other is not tx_cell:
                interf += 10 ** (self.recv_power_dBm(other, rx_cell, band) / 10)
        noise = 10 ** (self.noise_floor / 10)
        sinr = sig / (interf + noise)
        sinr_dB = 10 * math.log10(sinr)
        return sinr_dB >= 10.0

class BaseStation:
    """
    Represents a base station for Wi-Fi or NR-U, serving associated cells.

    Attributes:
        user_count: number of attached cells
        global_share: estimated share of spectrum (set by runner)
    """
    def __init__(self, env: simpy.Environment,
                 channel: Channel,
                 name: str, tech: str,
                 position: tuple, band: str,
                  cw_min=15, cw_max=63, ed_threshold_dBm=None):
        self.nav_expiry_time = 0.0
        self.env = env
        self.channel = channel
        self.name = name
        self.tech = tech
        self.position = position
        self.band = band
        self.cw_min = cw_min    # for WiFi
        self.cw_max = cw_max    # for WiFi
        self.ed_threshold = ed_threshold_dBm if ed_threshold_dBm is not None else (
            -62 if tech == 'WiFi' else -72)
        # will be computed externally
        self.global_share = 1.0
        self.served_cells = []
        self.backoff_event = env.event()        # یک Event خالی برای broadcast
        self.cells = []
    def register(self, cell):
        self.cells.append(cell)
    @property
    def user_count(self) -> int:
        """Return the number of attached cells (users)."""
        return len(self.served_cells)

    def attach(self, cell):
        cell.base_station = self
        self.served_cells.append(cell)
    def monitor(self, interval=0.1, busy_threshold=3, nav_duration=0.005):
        busy_count = 0
        while True:
            idle = self.channel.is_idle(self.band, self, self.ed_threshold)
            if not idle:
                busy_count += 1
            else:
                busy_count = 0

         # [1] NAV برای WiFi مثل قبل
            if busy_count >= busy_threshold:
                # print(f"[{self.name}] Channel busy → issuing NAV")
                self.nav_expiry_time = self.env.now + nav_duration
                busy_count = 0

        # [2] NEW: Backoff warning grant (for NR-U)
            elif busy_count == busy_threshold - 1:
                # print(f"[{self.name}] Predicting congestion → issuing backoff grant")
                self.backoff_event.succeed()    # fire fairness‐based deferral
                self.backoff_event = self.env.event()     # event جدید بساز

            yield self.env.timeout(interval)
    def monitor_fairness(self, interval=1.0,factor=4, recent_tx_threshold=3):
        """
        Periodically check if any user is starved.
        We compute starvation_delay = factor × avg_delay_seconds at each step.
        """
        while True:
            # 1) compute dynamic threshold (in seconds)
            #    gather all recorded delays (in seconds) into one list
            all_delays = [
                d for delays in self.metrics.delay_records.values() for d in delays
            ]
            if all_delays:
                avg_delay = sum(all_delays) / len(all_delays)
            else:
                avg_delay = interval  # fall back to one slot if no data yet
            starvation_delay = avg_delay * factor

        # 2) detect starving and fast users using this threshold
            starved = []
            fast_users = []
            for cell in self.served_cells:
                delays = self.metrics.delay_records.get(cell.name, [])
                if not delays:
                    continue
                cell_avg = sum(delays) / len(delays)
                if cell_avg > starvation_delay:
                    starved.append(cell.name)
            # “fast” = much faster than avg_delay
                elif cell_avg < (avg_delay / factor) and cell.tx_count >= recent_tx_threshold:
                    fast_users.append(cell)

        # 3) if anyone truly starved, ask the fast ones to defer
            if starved and fast_users:
                # print(f"[{self.name}] Dynamic starvation threshold = {starvation_delay:.6f}s")
                # print(f"[{self.name}] Starving: {starved}; telling fast {', '.join(f.name for f in fast_users)} to defer")
                self.backoff_event.succeed()
                self.backoff_event = self.env.event()

        # 4) wait one slot before re-checking
            yield self.env.timeout(interval)