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
        # initialize WiFi CW
        if self.tech == 'WiFi':
            self.cw = self.base_station.cw_min
        while True:
           # زمان اسلات واقعی = پایه تقسیم بر سهم جهانی BS
            slot_time = base_slot / self.base_station.global_share
            # pick backoff slots
            if self.tech == 'WiFi':
                 raw_slots = random.randint(1, self.cw-1)
                 backoff_slots = max(1, int(raw_slots / self.priority_weight))
            else:
                raw = ca_decision(self, self.grid, T=4, delta=1)
                backoff_slots = max(1, int(raw / self.priority_weight))
            self.backoff_count += 1
            yield self.env.timeout(backoff_slots * slot_time)

            # sense channel with ED threshold
            ed = self.base_station.ed_threshold
            band = self.base_station.band
            if self.channel.is_idle(band, self, ed):
                # transmit
                pkt_duration = 1.0 / self.base_station.global_share
                self.channel.occupy(band, self, pkt_duration)
                self.state['last_tx'] = True
                self.tx_count += 1
                print(f"{self.env.now:.6f}: {self.name} TX#{self.tx_count} on {band}")
                yield self.env.timeout(pkt_duration)
                self.channel.release(band, self)
                # reset CW on success
                if self.tech == 'WiFi':
                    self.cw = self.base_station.cw_min
                # add new traffic
                arrivals = self.generate_traffic()
                self.state['queue_len'] += arrivals
            else:
                # busy or collision
                self.state['last_tx'] = False
                if self.tech == 'WiFi':
                    self.cw = min(self.cw * 2, self.base_station.cw_max)
                # loop again
                continue
