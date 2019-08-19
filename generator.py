import json

class Generator:
    def __init__(self, flow_file, signal_file):
        with open(flow_file) as f:
            l_flow = sum(1 for line in f)
        with open(signal_file) as f:
            l_signal = sum(1 for line in f)
        assert l_flow == l_signal, "flow and signal has different time step"
        self.steps = l_flow
        self.f_flow = open(flow_file)
        self.f_signal = open(signal_file)
        self.read = False

    def action_required(self):
        self.flows = json.loads(self.f_flow.readline())
        self.signals = json.loads(self.f_signal.readline())
        self.read = True
        return len(self.flows) > 0

    def step(self, eng, params):
        if not self.read:
            self.flows = json.loads(self.f_flow.readline())
            self.signals = json.loads(self.f_signal.readline())
        for flow in self.flows:
            assert params is not None
            eng.push_vehicle(params, flow)
        for intersection, phase in self.signals.items():
            eng.set_tl_phase(intersection, phase)
        self.read = False

    def reset(self):
        self.f_flow.seek(0)
        self.f_signal.seek(0)
        self.read = False

    def __del__(self):
        self.f_flow.close()
        self.f_signal.close()