import argparse
import json
from generate_json_from_grid import gridToRoadnet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rowNum", type=int)
    parser.add_argument("colNum", type=int)
    parser.add_argument("--rowDistance", type=int, default=200)
    parser.add_argument("--columnDistance", type=int, default=200)
    parser.add_argument("--intersectionWidth", type=int, default=8)
    parser.add_argument("--numLeftLanes", type=int, default=0)
    parser.add_argument("--numStraightLanes", type=int, default=1)
    parser.add_argument("--numRightLanes", type=int, default=0)
    parser.add_argument("--laneMaxSpeed", type=float, default=30.0)
    parser.add_argument("--turn", action="store_true")
    parser.add_argument("--tl8", action="store_true")
    parser.add_argument("--green", type=int, default=20)
    parser.add_argument("--yellow", type=int, default=5)
    parser.add_argument("--interval", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--dir", type=str, default="../data/")
    parser.add_argument("--roadnetFile", type=str)
    parser.add_argument("--flowFile", type=str)
    parser.add_argument("--signalFile", type=str)
    return parser.parse_args()

def generate_route(rowNum, colNum, turn=False):
    routes = []
    move = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_straight_route(start, direction, step):
        x, y = start
        route = []
        for _ in range(step):
            route.append("road_%d_%d_%d" % (x, y, direction))
            x += move[direction][0]
            y += move[direction][1]
        return route

    for i in range(1, rowNum+1):
        routes.append(get_straight_route((0, i), 0, colNum+1))
        routes.append(get_straight_route((colNum+1, i), 2, colNum+1))
    for i in range(1, colNum+1):
        routes.append(get_straight_route((i, 0), 1, rowNum+1))
        routes.append(get_straight_route((i, rowNum+1), 3, rowNum+1))
    
    if turn:
        def get_turn_route(start, direction):
            if direction[0] % 2 == 0:
                step = min(rowNum*2, colNum*2+1)
            else:
                step = min(colNum*2, rowNum*2+1)
            x, y = start
            route = []
            cur = 0
            for _ in range(step):
                route.append("road_%d_%d_%d" % (x, y, direction[cur]))
                x += move[direction[cur]][0]
                y += move[direction[cur]][1]
                cur = 1 - cur
            return route

        routes.append(get_turn_route((1, 0), (1, 0)))
        routes.append(get_turn_route((0, 1), (0, 1)))
        routes.append(get_turn_route((colNum+1, rowNum), (2, 3)))
        routes.append(get_turn_route((colNum, rowNum+1), (3, 2)))
        routes.append(get_turn_route((0, rowNum), (0, 3)))
        routes.append(get_turn_route((1, rowNum+1), (3, 0)))
        routes.append(get_turn_route((colNum+1, 1), (2, 1)))
        routes.append(get_turn_route((colNum, 0), (1, 2)))
    
    return routes

if __name__ == '__main__':
    args = parse_args()
    if args.roadnetFile is None:
        args.roadnetFile = "roadnet_%d_%d%s.json" % (args.rowNum, args.colNum, "_turn" if args.turn else "")
    if args.flowFile is None:
        args.flowFile = "flow_%d_%d%s.jsonl" % (args.rowNum, args.colNum, "_turn" if args.turn else "")
    if args.signalFile is None:
        args.signalFile = "signal_%d_%d%s.jsonl" % (args.rowNum, args.colNum, "_turn" if args.turn else "")

    # generate roadnet file
    grid = {
        "rowNumber": args.rowNum,
        "columnNumber": args.colNum,
        "rowDistances": [args.rowDistance] * (args.colNum-1),
        "columnDistances": [args.columnDistance] * (args.rowNum-1),
        "outRowDistance": args.rowDistance,
        "outColumnDistance": args.columnDistance,
        "intersectionWidths": [[args.intersectionWidth] * args.colNum] * args.rowNum,
        "numLeftLanes": args.numLeftLanes,
        "numStraightLanes": args.numStraightLanes,
        "numRightLanes": args.numRightLanes,
        "laneMaxSpeed": args.laneMaxSpeed,
        "tlPlan": not args.tl8
    }
    roadnet = gridToRoadnet(**grid)
    json.dump(roadnet, open(args.dir + args.roadnetFile, "w"), indent=2)

    # generate flow file
    routes = generate_route(args.rowNum, args.colNum, args.turn)
    with open(args.dir + args.flowFile, "w") as f:
        for t in range(args.steps):
            flow = []
            if t % args.interval == 0:
                for route in routes:
                    flow.append(route)
            f.write(json.dumps(flow, separators=(',', ':')) + "\n")

    # generate signal plan
    intersections = []
    for intersection in roadnet["intersections"]:
        if not intersection["virtual"]:
            intersections.append(intersection["id"])

    if args.turn:
        plan = [[args.green, 0], [args.yellow, 1], [args.green, 2], [args.yellow, 3],
                [args.green, 4], [args.yellow, 5], [args.green, 6], [args.yellow, 7]]
    else:
        plan = [[args.green, 0], [args.yellow, 1], [args.green, 2], [args.yellow, 3]]
    elapsed = args.yellow
    curphase = len(plan) - 1
    with open(args.dir + args.signalFile, "w") as f:
        for t in range(args.steps):
            signals = {}
            if elapsed >= plan[curphase][0]:
                curphase = (curphase + 1) % len(plan)
                for intersection in intersections:
                    signals[intersection] = plan[curphase][1]
                elapsed = 0
            elapsed += 1.0
            f.write(json.dumps(signals, separators=(',', ':')) + "\n")
            
    

