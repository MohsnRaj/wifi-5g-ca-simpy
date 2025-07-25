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
                 grid=None,priority_weight: float = 1.0,cw_min=16,cw_max=1024, traffic_model="satured", lam=50,
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
        self.tx_count = 0
        self.backoff_count = 0
        self.state = {'queue_len':0, 'last_tx':False}
        # ─── EDCA-based CW/AIFS for Wi-Fi ───────────────────────────────────────
        if tech == "WiFi":
            # map primary users → AC_VO (highest priority)
            if self.priority_weight >= 1.5:
                self.ac          = "AC_VO"
                self.cw_min      = 3       # CWmin for voice
                self.cw_max      = 7       # CWmax for voice
                self.aifs_slots  = 2       # AIFS for voice
                self.txop_limit  = 0.002   # 2 ms TXOP
            # map secondary users → AC_BE (best effort)
            else:
                self.ac          = "AC_BE"
                self.cw_min      = 15      # CWmin for best-effort
                self.cw_max      = 1023    # CWmax for best-effort
                self.aifs_slots  = 6       # AIFS for best-effort
                self.txop_limit  = 0.002   # 2 ms TXOP
        else:
            # NR-U (LAA-style)
            if self.priority_weight >= 1.5:
                self.ac          = "NRU_High"
                self.cw_min      = 7
                self.cw_max      = 63
                self.aifs_slots  = 1
                self.txop_limit  = 0.001  # 1 ms
            else:
                self.ac          = "NRU_Low"
                self.cw_min      = 15
                self.cw_max      = 1023
                self.aifs_slots  = 4
                self.txop_limit  = 0.0005  # 0.5 ms

        # pick initial CW in our new [cw_min…cw_max]
        self.cw = random.randint(self.cw_min, self.cw_max)
        self.T_min     = 2.0
        self.T_max     = 8.0  
        self.T_dynamic = random.uniform(2, 5)
        self.last_tx_time = 0.0
        self.defer_time = 0.0
        self.slot_time = 0.0
        # --- AIMD and EMA control parameters ---
        # Initialize these once before the while loop:
        if tech=="NR-U":
            self.alpha_gap = 0.9    # smoother gap estimate
            self.dec_factor= 0.7    # less drastic cuts when congested
            self.inc_step  = 0.3    # slower growth when underutilized
        else:  # Wi-Fi
            self.alpha_gap = 0.8
            self.dec_factor= 0.5
            self.inc_step  = 0.5    
        if phy_rate_bps is None:
            self.phy_rate_bps = 54e6 if tech=='WiFi' else 100e6
        else:
            self.phy_rate_bps = phy_rate_bps
        Cell.registry.append(self)
        if self.traffic_model == "poisson":
            nw_lam = random.uniform(self.traffic_lambda * 0.8,
                                 self.traffic_lambda * 1.2)
            self.env.process(self.traffic_generator_poisson(lam=nw_lam))
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
            Saturated traffic—but only one outstanding packet at a time.
            After the PHY airtime elapses (±5% jitter), we refill only
            when the queue is empty, so you never build up a backlog.
            """
            payload_bytes = random.randint(1200, 1500)  
            packet_bits   = payload_bytes * 8  

            # wait until BS share is known
            while not hasattr(self, 'base_station'):
                yield self.env.timeout(1e-6)

            # compute average airtime per packet
            avg_pkt_time = packet_bits / (self.phy_rate_bps * self.base_station.global_share)

            # seed the very first packet
            self.queue.append(self.env.now)

            while True:
                # if there's still a packet waiting, hold off any new arrivals
                if self.queue:
                    yield self.env.timeout(1e-6)
                    continue

                # once queue is empty, wait for the next arrival (±5% jitter)
                inter_arrival = random.uniform(0.8 * avg_pkt_time,
                                               1.2 * avg_pkt_time)
                yield self.env.timeout(inter_arrival)

                # now that queue is empty, enqueue one packet
                self.queue.append(self.env.now)

    def neighbor_starvation_detected(self, delay_threshold=0.001):
        
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
        self.slot_time =slot_time
        target_gap = self.T_dynamic * slot_time
        self.gap_avg = target_gap
        packet_bits = 1500 * 8
        share     = self.base_station.global_share
        avg_pkt_time = packet_bits / (self.phy_rate_bps * share)
        # small initial jitter (0–10% of desired gap)
        yield self.env.timeout(random.uniform(0, avg_pkt_time))
        while True:
            # --- compute EMA of transmission gap ---
            measured_gap = self.env.now - self.last_tx_time
            # EMA update: gap_avg = alpha * gap_avg + (1 - alpha) * measured_gap
            self.gap_avg = (self.alpha_gap * self.gap_avg
                            + (1 - self.alpha_gap) * measured_gap)

            # --- update T_dynamic using AIMD ---
            # Compare EMA gap with the target gap
            if self.gap_avg < target_gap:
                # congested: multiplicative decrease
                self.T_dynamic = max(self.T_dynamic * self.dec_factor,
                                    self.T_min)
            else:
                # underutilized: additive increase
                self.T_dynamic = min(self.T_dynamic + self.inc_step,
                                    self.T_max)

            # --- recalculate target_gap after updating T_dynamic ---
            slot_time  = base_slot / self.base_station.global_share
            target_gap = self.T_dynamic * slot_time

            # use the new T_dynamic everywhere
            penalty_slots, defer_strength = ca_decision(self, self.grid, T=self.T_dynamic)
            # defer_strength = self.neighbor_starvation_detected()

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
            if penalty_slots > 0:
                # we got deferred because neighbors outnumber our window → loosen
                self.T_dynamic = min(self.T_dynamic + random.uniform(0.1, 0.5), 8)
            else:
                # we were allowed immediately → tighten for more fairness
                self.T_dynamic = max(self.T_dynamic - random.uniform(0.1, 0.5), 2)
            # 1) pick the normal CSMA/LBT backoff
            raw_slots = random.randint(0, self.cw - 1)

            
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
            aifs_wait = self.aifs_slots * slot_time + random.uniform(0, slot_time)
            yield self.env.timeout(aifs_wait)
        # Channel sensing
            ed = getattr(self, 'ed_threshold', self.base_station.ed_threshold)
            band = self.base_station.band
            if self.queue and self.channel.is_idle(band, self, ed):
                #Begin transmission
                packet_bits = 1500 * 8                  # bits per packet
                phy_rate_bps = self.phy_rate_bps                     
                share = self.base_station.global_share 

                pkt_duration = packet_bits / (phy_rate_bps * share)
                self.channel.occupy(band, self, pkt_duration)
            
                # Record delay for first packet
                
                delay = self.env.now - self.queue.pop(0)
                self.base_station.metrics.record_delay(self.name, delay)
    
                self.state['queue_len'] = len(self.queue)
                self.state['last_tx'] = True
                self.tx_count += 1
                yield self.env.timeout(pkt_duration)

                success = self.channel.can_receive(band, self, self.base_station)

                if success:
                    self.base_station.metrics.record_success(self.name)
                    self.last_tx_time = self.env.now
                    self.T_dynamic = max(self.T_dynamic - 0.5, self.T_min)
                # Reset CW on success
                    if self.tech == 'WiFi' or self.tech == 'NR-U':
                        self.cw = self.base_station.cw_min
                else:
                    self.base_station.metrics.record_loss(self.name)
                    self.T_dynamic = min(self.T_dynamic + 0.7, self.T_max)
                # Double CW on failure (if applicable)
                    if self.tech == 'WiFi' or self.tech == 'NR-U':
                        self.cw = min(self.cw * 2, self.base_station.cw_max)

            # Release channel
                self.channel.release(band, self)

            
                self.state['queue_len'] = len(self.queue)
                # 📍 OPTIONAL VOLUNTARY DEFER FOR NR-U SECONDARY
                if self.tech == "NR-U" and self.priority_weight < 1.5:
                    # Secondary NR-U voluntarily defers to improve fairness
                    skip_slots = random.randint(1, 3)
                    defer_time = skip_slots * slot_time
                    # print(f"[{self.name}] Skipping {skip_slots} slots voluntarily")
                    yield self.env.timeout(defer_time)
                continue  # go back to while loop
            if not self.queue:
                yield self.env.timeout(slot_time)            
            else:
            # Busy or collision occurred
                self.state['last_tx'] = False
                if self.tech == 'WiFi' or self.tech == 'NR-U':
                    self.cw = min(self.cw * 2, self.base_station.cw_max)
                continue
