import simpy
import random
import numpy as np
from .ca_rules import ca_decision

class Cell:
    """
    A network cell node in the SimPy-based CA simulator.
    Applies CSMA/CA (WiFi) or LBT (NR-U) + global share slot adjustment.
    """
    registry = []

    def __init__(self, env, name, tech, channel, position, model=None, grid=None,priority_weight: float = 1.0):
        self.env = env
        self.queue = []
        self.name = name
        self.tech = tech
        self.channel = channel
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
        Cell.registry.append(self)

    def just_transmitted(self):
        return self.state.get('last_tx', False)

    def generate_traffic(self):
        return np.random.poisson(lam=0.5)

    def run(self):
        
        assert hasattr(self, 'base_station'), \
            f"Cell {self.name} must be attached to a BaseStation before run()"

    # base slot duration per technology
        base_slot = 0.001 if self.tech=='NR-U' else 0.009
        slot_time = base_slot / self.base_station.global_share
        target_gap = self.T_dynamic * slot_time
        if self.tech == 'NR-U':
            self.cw = self.base_station.cw_min
    # initialize CW based on technology
        if self.tech == 'WiFi' or self.tech == 'NR-U':
            self.cw = self.base_station.cw_min

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
            
            print(f"[DEBUG] {self.name} → newT={self.T_dynamic}", flush=True)

            # use the new T_dynamic everywhere
            ca_raw = ca_decision(self, self.grid, T=self.T_dynamic, delta=1)
            if ca_raw > 0:
                # we got deferred because neighbors outnumber our window → loosen
                self.T_dynamic = min(self.T_dynamic +2, 8)
            else:
                # we were allowed immediately → tighten for more fairness
                self.T_dynamic = max(self.T_dynamic - 1, 2)
            print(f"[DEBUG] {self.name} adapted T to {self.T_dynamic} based on CA")
            if ca_raw == 0:
                raw_slots    = random.randint(1, self.cw - 1)
                backoff_slots = max(1, int(raw_slots / self.priority_weight))
            else:
                backoff_slots = ca_raw
        # Total backoff time in simulation
            slot_time = base_slot / self.base_station.global_share
            backoff_timeout = self.env.timeout(backoff_slots * slot_time)
            bs_backoff_evt = self.base_station.backoff_event

        # Count number of backoff attempts
            self.backoff_count += 1

        # Wait until either backoff timeout or BS signal to stop
            result = yield backoff_timeout | bs_backoff_evt

            if bs_backoff_evt in result:
                continue

        # Check NAV before proceeding (WiFi only)
            if self.tech == 'WiFi' and self.env.now < self.base_station.nav_expiry_time:
                nav_wait = self.base_station.nav_expiry_time - self.env.now
                print(f"{self.name} sees NAV active → waiting {nav_wait:.4f}s")
                yield self.env.timeout(nav_wait)

        # Channel sensing
            ed = self.base_station.ed_threshold
            band = self.base_station.band
            if self.channel.is_idle(band, self, ed):
                # Begin transmission
                pkt_duration = 1.0 / self.base_station.global_share
                self.channel.occupy(band, self, pkt_duration)

            # Record delay for first packet
                if self.queue:
                    delay = self.env.now - self.queue.pop(0)
                    self.base_station.metrics.record_delay(self.name, delay)

                self.state['queue_len'] = len(self.queue)
                self.state['last_tx'] = True
                self.tx_count += 1
                print(f"{self.env.now:.6f}: {self.name} TX#{self.tx_count} on {band}")
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

            # Add traffic to queue
                arrivals = self.generate_traffic()
                for _ in range(arrivals):
                    self.queue.append(self.env.now)
                self.state['queue_len'] = len(self.queue)

                continue  # go back to while loop

            else:
            # Busy or collision occurred
                self.state['last_tx'] = False
                if self.tech == 'WiFi' or self.tech == 'NR-U':
                    self.cw = min(self.cw * 2, self.base_station.cw_max)
                continue
