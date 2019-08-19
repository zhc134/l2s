import numpy as np
import collections
import math

class Processor():
    
    def __init__(self):
        pass
        
    def process_line(self, line):
        v = {}
        fields = line.strip().strip(',').split(',')
        for field in fields:
            entries = field.strip().split(' ')
            field_name = entries[0]
            v[field_name] = {}
            v_field = v[field_name]
            if field_name == "lane_speed":
                for entry in entries[1:]:
                    eid, val = entry.split(':')
                    v_field[eid] = float(val)
        return v

    def get_output(self, eng, string=False):
        lanes = eng.get_lane_vehicles()
        speed = eng.get_vehicle_speed()

        if string:
            ret = "lane_speed "
            ret += " ".join(["%s:%.6f" % (
                                lane,
                                np.mean([speed[veh] for veh in lanes[lane]]) if len(lanes[lane]) else 0
                            ) for lane in lanes])
        else:
            ret = {
                "lane_speed": dict([(lane, np.mean([speed[veh] for veh in lanes[lane]]) 
                                    if len(lanes[lane]) else 0) for lane in lanes])
            }
        return ret

    def get_loss(self, generated, observed):
        if isinstance(generated, str):
            generated = self.process_line(generated)
        if isinstance(observed, str):
            observed = self.process_line(observed)

        loss = {}
        for field in observed:
            vO = observed[field]
            vG = generated[field]
            ret = dict()
            for entry in vO:
                if math.isclose(vO[entry], 0) and math.isclose(vG[entry], 0):
                    ret[entry] = 0
                else:
                    ret[entry] = (vG[entry] - vO[entry]) / max(vG[entry], vO[entry])
            loss[field] = ret
        return loss

    def get_reward(self, generated, observed):
        loss = self.get_loss(generated, observed)
        reward = 0
        for field in loss:
            loss_ = 0
            field_loss = loss[field]
            for entry in field_loss:
                loss_ += abs(field_loss[entry])
            reward += -(loss_ / len(field_loss))
        reward /= len(loss)
        return reward

    def get_state(self, generated, observed):
        loss = self.get_loss(generated, observed)
        state = []
        for field in loss:
            field_loss = loss[field]
            field_loss_vector = np.array([
                field_loss[entry] for entry in sorted(field_loss.keys())
            ])
            state.append(np.mean(field_loss_vector))
        return np.array(state)

if __name__ == '__main__':
    pass