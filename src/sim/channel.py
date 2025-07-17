import simpy

class Channel:
    def __init__(self, env):
        self.busy = simpy.Container(env, capacity=1, init=0)

    def is_idle(self):
        return self.busy.level == 0

    def occupy(self):
        return self.busy.put(1)

    def release(self):
        return self.busy.get(1)
