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
        self.my_delays = []     # list of delay samples
        self.my_avg_delay = 0.0
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
        self.neighbor_info = {}  # stores latest status of same-tech neighbors
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
                self.cw_max      = 127
                self.aifs_slots  = 1
                self.txop_limit  = 0.002

        # pick initial CW in our new [cw_min…cw_max]
        self.cw = random.randint(self.cw_min, self.cw_max)
        self.T_min     = 2.0
        self.T_max     = 8.0  
        self.T_dynamic = random.uniform(2, 5)
        self.gap_avg = 0.0
        self.last_tx_time = 0.0
        self.defer_time = 0.0
        self.slot_time = 0.0
        # --- AIMD and EMA control parameters ---
        # Initialize these once before the while loop:
        if tech=="NR-U":
            self.alpha_gap = 0.9    # smoother gap estimate
            self.dec_factor= 0.8    # less drastic cuts when congested
            self.inc_step  = 0.2    # slower growth when underutilized
        else:  # Wi-Fi
            self.alpha_gap = 0.8
            self.dec_factor= 0.4
            self.inc_step  = 0.6    
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
        self.env.process(self.periodic_broadcast())
    
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
    def update_T_dynamic(self, defer_strength: int):
        """
    Adjust T_dynamic based on defer signals from neighbor_starvation_detected():
      - defer_strength = 0 → no change
      - defer_strength = 1 → mild defer
      - defer_strength = 2 → full defer
    Smaller steps for NR-U to avoid too-aggressive swings.
    """
        if defer_strength == 0:
            return  # nothing to do

        tech = self.tech       # "WiFi" or "NR-U"
        prio = self.ac         # "AC_VO", "AC_BE", "NRU_High", or "NRU_Low"

        # -------------------------------
        # 1) Wi-Fi logic (primaries & secondaries)
        # -------------------------------
        if tech == "WiFi":
            if prio == "AC_VO":
                # Voice (primary) ≔ gentle adjustment
                if defer_strength == 1:
                    # mild defer → increase T by +0.5 up to T_max
                    self.T_dynamic = min(self.T_dynamic + 0.5, self.T_max)
                else:
                    # full defer → shrink T by 20% but no lower than T_min
                    self.T_dynamic = max(self.T_dynamic * 0.8, self.T_min)

            else:  # AC_BE
                # Best-effort (secondary) ≔ bit stronger than voice
                if defer_strength == 1:
                    self.T_dynamic = min(self.T_dynamic + 0.8, self.T_max)
                else:
                    self.T_dynamic = max(self.T_dynamic * 0.8, self.T_min)

        # -------------------------------
        # 2) NR-U logic with softer steps
        # -------------------------------
        else:  # tech == "NR-U"
            if prio == "NRU_High":
                # High-priority NR-U ≔ small step increases
                if defer_strength == 1:
                    self.T_dynamic = min(self.T_dynamic + 0.5, self.T_max)
                else:
                    # reduce by just 5% on full defer
                    self.T_dynamic = max(self.T_dynamic * 0.95, self.T_min)

            else:  # NRU_Low
                # Low-priority NR-U ≔ very mild increase
                if defer_strength == 1:
                    self.T_dynamic = min(self.T_dynamic + 0.2, self.T_max)
                else:
                    # shrink by 10% on full defer
                    self.T_dynamic = max(self.T_dynamic * 0.9, self.T_min)

        # -------------------------------
        # 3) Optional extra check for low-weight NR-U
        # -------------------------------
        # If this NR-U BS is “lightweight” and its actual delay is still too high,
        # give it a small further reduction to catch up.
        if tech == "NR-U" and self.priority_weight < 1.5:
            delays = self.base_station.metrics.delay_records.get(self.name, [])
            if delays:
                avg_delay = sum(delays) / len(delays)
                if avg_delay > 0.0005:
                    # trim another 10%
                    self.T_dynamic = max(self.T_dynamic * 0.9, self.T_min)

    def broadcast_status(self):
        """
        Periodically broadcast this cell's MAC-level status to same-tech neighbors.
        """
        msg = {
            "sender": self.name,
            "priority": self.priority_weight,
            "avg_delay": getattr(self, 'my_avg_delay', 0.0),
            "cw": self.cw,
            "T": self.T_dynamic,
            "last_tx": self.last_tx_time,
            "position": self.position,
        }
        msg['tx_count'] = self.tx_count
        x, y = self.position
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = self.grid.get((x + dx, y + dy))
                # send only to same-technology neighbors
                if neighbor and neighbor.tech == self.tech:
                    neighbor.receive_status(msg)
    def receive_status(self, msg):
        """
        Store received status from a same-tech neighbor.
        """
        self.neighbor_info[msg['sender']] = msg
    def periodic_broadcast(self, interval=0.001):
        """
        Run broadcast_status() every 'interval' seconds.
        """
        while True:
            self.broadcast_status()
            yield self.env.timeout(interval)
    def neighbor_starvation_detected(self, delay_threshold=0.001):
        
        """Check delay of neighbors, return defer strength: 0 (none), 1 (mild), 2 (strong)"""
        if self.tech == "NR-U":
            delay_threshold *= 0.95
        
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
                        if neighbor.tech == "NR-U":
                            avg_delay *= 1.4
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

        # ۱) پارامترهای اولیه
        base_slot     = 25e-6 if self.tech=='NR-U' else 9e-6
        packet_bits   = 1500 * 8
        share         = self.base_station.global_share
        avg_pkt_time  = packet_bits / (self.phy_rate_bps * share)

        # ۲) jitter اولیه
        yield self.env.timeout(random.uniform(0, avg_pkt_time))

        # ۳) صبر یک دوره broadcast تا neighbor_info پر بشه
        yield self.env.timeout(0.0001)
        # self.last_tx_time = self.env.now


        # ۴) حلقه اصلی
        while True:
            
            slot_time  = base_slot / self.base_station.global_share
            if self.tx_count > 0:
                # ——— EMA فاصله ارسال‌ها ———
                measured_gap = self.env.now - self.last_tx_time
                slot_time   = base_slot / self.base_station.global_share
                gap_in_slots = (measured_gap / slot_time) /100
                self.gap_avg = (self.alpha_gap * self.gap_avg
                                + (1 - self.alpha_gap) * gap_in_slots)

                target_gap = self.T_dynamic

                
                total_bs, count = 0.0, 0
                for info in self.neighbor_info.values():
                    age = self.env.now - info['last_tx']
                    if age <= 0.005:
                        total_bs += info['tx_count']
                        count += 1
                neighbor_busy_avg = total_bs / count if count else 0.0

                # ——— دینامیک CW براساس busy_score ———
                if neighbor_busy_avg > target_gap:
                    self.cw = int(min(self.cw * 1.1, self.cw_max))
                elif neighbor_busy_avg < 0.5 * target_gap:
                    self.cw = int(max(self.cw * 0.9, self.cw_min))

               
            # ——— تصمیم CA با مقدار جدید T_dynamic ———
            penalty_slots, defer_strength = ca_decision(self, self.grid,
                                                    T=self.T_dynamic)
            self.update_T_dynamic(defer_strength)

            # —— بقیه‌ی logic backoff و ارسال ——
            if defer_strength == 2:
                # Secondary full defer
                yield self.env.timeout(2 * slot_time)
                continue
            elif defer_strength == 1:
                # Primary mild defer
                yield self.env.timeout(1 * slot_time)
            # ادامه به CA logic

            # ۱) انتخاب backoff slots
            raw_slots = (int(random.uniform(0, self.cw/2)) 
                        if self.tech=="NR-U" else
                        random.randint(0, self.cw-1))
            total_slots = max(1, raw_slots + penalty_slots)

            # ۲) backoff timeout
            backoff_timeout = self.env.timeout(total_slots * slot_time)
            result = yield backoff_timeout | self.base_station.backoff_event

            # فقط اگر NR-U بود، از دستور BS اطاعت کن
            if self.tech == 'NR-U' and self.base_station.backoff_event in result:
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
                self.my_delays.append(delay)
                alpha = 0.8
                self.my_avg_delay = alpha * self.my_avg_delay + (1 - alpha) * delay
                self.state['queue_len'] = len(self.queue)
                self.state['last_tx'] = True
                self.tx_count += 1
                yield self.env.timeout(pkt_duration)

                success = self.channel.can_receive(band, self, self.base_station)

                if success:
                    self.base_station.metrics.record_success(self.name)
                    self.last_tx_time = self.env.now
                    # self.T_dynamic = max(self.T_dynamic - 0.5, self.T_min)
                # Reset CW on success
                    if self.tech == 'WiFi' or self.tech == 'NR-U':
                        self.cw = self.base_station.cw_min
                else:
                    self.base_station.metrics.record_loss(self.name)
                    # self.T_dynamic = min(self.T_dynamic + 0.7, self.T_max)
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
