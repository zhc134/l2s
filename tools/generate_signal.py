import argparse
import json
import pandas as pd
from generate_json_from_grid import gridToRoadnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowNum", type=int, default=1)
    parser.add_argument("--colNum", type=int, default=3)
    parser.add_argument("--rowDistance", type=int, default=300)
    parser.add_argument("--columnDistance", type=int, default=300)
    parser.add_argument("--intersectionWidth", type=int, default=30)
    parser.add_argument("--numLeftLanes", type=int, default=1)
    parser.add_argument("--numStraightLanes", type=int, default=1)
    parser.add_argument("--numRightLanes", type=int, default=1)
    parser.add_argument("--laneMaxSpeed", type=float, default=16.67)
    parser.add_argument("--vehLen", type=float, default=5.0)
    parser.add_argument("--vehWidth", type=float, default=2.0)
    parser.add_argument("--vehMaxPosAcc", type=float, default=2.0)
    parser.add_argument("--vehMaxNegAcc", type=float, default=4.5)
    parser.add_argument("--vehUsualPosAcc", type=float, default=2.0)
    parser.add_argument("--vehUsualNegAcc", type=float, default=4.5)
    parser.add_argument("--vehMinGap", type=float, default=2.5)
    parser.add_argument("--vehMaxSpeed", type=float, default=16.67)
    parser.add_argument("--vehHeadwayTime", type=float, default=1.5)
    parser.add_argument("--dir", type=str, default="../data/")
    parser.add_argument("--roadnetFile", type=str)
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--tlPlan", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--flowFile", type=str)
    parser.add_argument("--simulationTime", type=int, default=3600)
    parser.add_argument("--numPhase", type=int, default=2)
    return parser.parse_args()

def generate_inter_list(grid_row_num, grid_column_num):

    inters = []
    for i in range(1, grid_column_num+1):
        for j in range(1, grid_row_num+1):
            inters.append("intersection_{0}_{1}".format(i, j))
    return inters


def generate_signals(num_phase, simulation_time, interval, inters):

    signals_head = ["time"] + inters
    signals = []
    phases = [1, 2]

    phase_index = 0
    yellow = 5
    for t in range(simulation_time):
        if t % (interval + yellow) == 0:
            phase_index = (phase_index + 1) % num_phase
            phase = phases[phase_index]
        elif t % (interval + yellow) == interval:
            phase = 0
        signal_at_t = [t]
        for inter in inters:
            signal_at_t.append(phase)
        signals.append(signal_at_t)

    return pd.DataFrame(signals, columns=signals_head)


if __name__ == '__main__':
    args = parse_args()
    if args.flowFile is None:
        args.flowFile = "signal_%d_%d%s.json" % (args.rowNum, args.colNum, "_turn" if args.turn else "")

    inters = generate_inter_list(args.rowNum, args.colNum)
    signals = generate_signals(args.numPhase, args.simulationTime, args.interval, inters)
    signals.to_json(args.dir + "signal/" + args.flowFile)

