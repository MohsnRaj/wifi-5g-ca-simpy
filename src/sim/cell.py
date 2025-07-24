import random
import numpy as np
from .ca_rules import ca_decision

class Cell:
    """
    A network cell node in the SimPy-based CA simulator.
    Applies CSMA/CA (WiFi) or LBT (NR-U) + global share slot adjustment.
    """
    registry = []

    def __init__(self, env, name, tech, channel, position, model=None,
                 grid=None,priority_weight: float = 1.0, traffic_model="saturade", lam=50,
                 phy_rate_bps: float = None, queue_limit=100):
        self.env = env
        self.queue_limit = queue_limit
        self.queue = []
        self.name = name
        self.tech = tech
        self.channel = channel
        self.traffic_model = traffic_model
        self.traffic_lambda = lam
        self.position = position
        self.grid = grid
        self.model = model
        self.priority_weight = priority_weight
        self.cw = None
        self.tx_count = 0
        self.backoff_count = 0
        self.state = {'queue_len':0, 'last_tx':False}
        self.cw = None
        self.T_dynamic = 4  # Starting value (can go up or down)
        self.last_tx_time = 0.0
        if phy_rate_bps is None:
            self.phy_rate_bps = 54e6 if tech=='WiFi' else 100e6
        else:
            self.phy_rate_bps = phy_rate_bps
        Cell.registry.append(self)
        if self.traffic_model == "poisson":
            self.env.process(self.traffic_generator_poisson(lam=self.traffic_lambda))
        else:
            self.env.process(self.traffic_generator_saturated())

    def just_transmitted(self, window):
        # only count transmissions in the last `window` seconds
        return (self.env.now - self.last_tx_time) <= window
    def traffic_generator_poisson(self, lam=100):
        """
        Realistic traffic model based on Poisson arrivals.
        Generates packets with exponential inter-arrival time.
    
        Args:
            lam (float): Average packet arrival rate (packets per second).
        """
        while True:
            # Wait for the next packet arrival
            inter_arrival = np.random.exponential(1 / lam)
            yield self.env.timeout(inter_arrival)

            # Only add packet if buffer not full
            if len(self.queue) < self.queue_limit:
                self.queue.append(self.env.now)
    def traffic_generator_saturated(self):
        """
        Full-buffer (saturated) traffic: as soon as one packet is 'sent'
        we immediately enqueue the next, so the MAC is never idle.
        """
        # Preload one packet at t=0
        self.queue.append(self.env.now)
        while True:
           # as soon as the queue empties, refill instantly
            yield self.env.timeout(1e-6)  
            if len(self.queue) == 0:
                self.queue.append(self.env.now)
    # def generate_traffic(self):
    #     return np.random.poisson(lam=0.1)
    def neighbor_starvation_detected(self, delay_threshold=0.4):
        
        """Check delay of neighbors, return defer strength: 0 (none), 1 (mild), 2 (strong)"""
        if self.grid is None or self.base_station is None:
            return 0
        metrics = self.base_station.metrics
        x, y = self.position

        max_neighbor_delay = 0
        starving_neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_pos = (x + dx, y + dy)
                neighbor = self.grid.get(neighbor_pos)
                if neighbor:
                    delays = metrics.delay_records.get(neighbor.name, [])
                    if delays:
                        avg_delay = sum(delays)/len(delays)
                        if avg_delay > delay_threshold:
                            starving_neighbors.append((neighbor.name, avg_delay))
                            max_neighbor_delay = max(max_neighbor_delay, avg_delay)

        if not starving_neighbors:
            return 0

        # print(f"[{self.name}] detects starving neighbors: {starving_neighbors}")

        # Strong defer for secondary users
        if self.priority_weight < 1.5:
            return 2

        # Mild defer for primary users to help lower-priority starving neighbors
        return 1

    def run(self):
        
        assert hasattr(self, 'base_station'), \
            f"Cell {self.name} must be attached to a BaseStation before run()"

    # base slot duration per technology
        base_slot = 25e-6 if self.tech=='NR-U' else 9e-6
        slot_time = base_slot / self.base_station.global_share
        target_gap = self.T_dynamic * slot_time
        if self.tech == 'NR-U':
            self.cw = self.base_station.cw_min
    # initialize CW based on technology
        if self.tech == 'WiFi' or self.tech == 'NR-U':
            self.cw = self.base_station.cw_min
        # print(f"[DEBUG] {self.name}: Queue len = {len(self.queue)} at t={self.env.now}")
        while True:
            # ——— unified adaptive-T + CA decision ———
            # — debug — before adaptive step —
            time_since_last_tx = self.env.now - self.last_tx_time
            
            if time_since_last_tx <  target_gap:
    # you’re transmitting faster than your window, so tighten
                self.T_dynamic = max(self.T_dynamic - 1, 2)
            elif time_since_last_tx >  target_gap:
    # you’re starved for more than your own window, so loosen
                self.T_dynamic = min(self.T_dynamic + 1, 8)
            
            # print(f"[DEBUG] {self.name} → newT={self.T_dynamic}", flush=True)

            # use the new T_dynamic everywhere
            ca_raw = ca_decision(self, self.grid,
                     T=self.T_dynamic,
                     delta=1)
            defer_strength = self.neighbor_starvation_detected()

            if defer_strength == 2:
            # Secondary user → full defer
                defer_time = 2 * (base_slot / self.base_station.global_share)
                # print(f"[{self.name}] (S) full defer due to neighbor starvation")
                yield self.env.timeout(defer_time)
                continue
            elif defer_strength == 1:
            # Primary user → mild defer
                defer_time = 1 * (base_slot / self.base_station.global_share)
                # print(f"[{self.name}] (P) mild defer to help starved neighbor")
                yield self.env.timeout(defer_time)
            # not continue → continue with CA-decision
            if ca_raw > 0:
                # we got deferred because neighbors outnumber our window → loosen
                self.T_dynamic = min(self.T_dynamic +2, 8)
            else:
                # we were allowed immediately → tighten for more fairness
                self.T_dynamic = max(self.T_dynamic - 1, 2)
            # 1) pick the normal CSMA/LBT backoff
            raw_slots = random.randint(0, self.cw - 1)

            # 2) compute the CA‐penalty
            penalty_slots = ca_decision(self, self.grid,
                                        T=self.T_dynamic,
                                        delta=1)

            # 3) combine them (ensure at least one slot)
            total_slots = max(1, raw_slots + penalty_slots)

            # 4) convert to time
            slot_time = base_slot / self.base_station.global_share
            backoff_timeout = self.env.timeout(total_slots * slot_time)
            bs_backoff_evt = self.base_station.backoff_event

            # 5) count it
            self.backoff_count += 1

            # 6) wait for backoff *or* BS grant
            result = yield backoff_timeout | bs_backoff_evt

            # فقط اگر NR-U بود، از دستور BS اطاعت کن
            if self.tech == 'NR-U' and bs_backoff_evt in result:
                # print(f"[{self.name}] Deferred by BS grant (Type-4 style)")
                continue

        # Check NAV before proceeding (WiFi only)
            if self.tech == 'WiFi' and self.env.now < self.base_station.nav_expiry_time:
                nav_wait = self.base_station.nav_expiry_time - self.env.now
                # print(f"{self.name} sees NAV active → waiting {nav_wait:.4f}s")
                yield self.env.timeout(nav_wait)

        # Channel sensing
            ed = self.base_station.ed_threshold
            band = self.base_station.band
            if self.queue and self.channel.is_idle(band, self, ed):
                #Begin transmission
                packet_bits = 1500 * 8                  # bits per packet
                phy_rate_bps = self.phy_rate_bps                     
                share = self.base_station.global_share  # سهم شما از این نرخ

                pkt_duration = packet_bits / (phy_rate_bps * share)
                self.channel.occupy(band, self, pkt_duration)
            
            # Record delay for first packet
                
                delay = self.env.now - self.queue.pop(0)
                self.base_station.metrics.record_delay(self.name, delay)
    
                self.state['queue_len'] = len(self.queue)
                self.state['last_tx'] = True
                self.tx_count += 1
                # print(f"{self.env.now:.6f}: {self.name} TX#{self.tx_count} on {band}")
                yield self.env.timeout(pkt_duration)

                success = self.channel.can_receive(band, self, self.base_station)

                if success:
                    self.base_station.metrics.record_success(self.name)
                    self.last_tx_time = self.env.now
                # Reset CW on success
                    if self.tech == 'WiFi' or self.tech == 'NR-U':
                        self.cw = self.base_station.cw_min
                else:
                    self.base_station.metrics.record_loss(self.name)
                # Double CW on failure (if applicable)
                    if self.tech == 'WiFi' or self.tech == 'NR-U':
                        self.cw = min(self.cw * 2, self.base_station.cw_max)

            # Release channel
                self.channel.release(band, self)

            
                self.state['queue_len'] = len(self.queue)

                continue  # go back to while loop
            if not self.queue:
                yield self.env.timeout(slot_time)            
            else:
            # Busy or collision occurred
                self.state['last_tx'] = False
                if self.tech == 'WiFi' or self.tech == 'NR-U':
                    self.cw = min(self.cw * 2, self.base_station.cw_max)
                continue
