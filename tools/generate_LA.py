from generate_json_from_grid import findPath
import json
import numpy as np


path = '../data/setting/LA/'

moves = [(1,0), (0,1), (-1,0), (0,-1)]
typemap = {"l": "turn_left", "s": "go_straight", "r": "turn_right"}
turnmap = {"l": 1, "s": 0, "r": -1}

LANE_WIDTH = 4
LANE_MAX_SPEED = 16.67

NMIDPOINTS = 5
BRANCH_LENGTH = 50

INTER_WIDTH = 30

lane_setting = json.load(open(path + 'lane_setting.json'))
raw_intersections = json.load(open(path + 'inter_coor.json'))
phase_setting = json.load(open(path + 'phase_setting.json'))
for inter in raw_intersections:
    raw_intersections[inter] = {
        "point": dict(zip(["x", "y"], raw_intersections[inter])),
        "roadLinks": [],
        "trafficLight": {
            "lightphases": []
        },
        "virtual": False,
        "width": INTER_WIDTH
    }

intersections = raw_intersections.copy()
for inter in raw_intersections:
    _, x, y = inter.split('_')
    x = int(x)
    y = int(y)
    for i in range(4):
        inter2 = "intersection_%d_%d" % (x + moves[i][0], y + moves[i][1])
        if not inter2 in raw_intersections:
            intersections[inter2] = {
                "point": {
                    "x": raw_intersections[inter]["point"]["x"] + moves[i][0] * BRANCH_LENGTH,
                    "y": raw_intersections[inter]["point"]["y"] + moves[i][1] * BRANCH_LENGTH
                },
                "virtual": True,
                "width": 0
            }

roads = {}

for road in lane_setting:
    _, x, y, direction = road.split('_')
    x = int(x)
    y = int(y)
    direction = int(direction)

    roads[road] = {
        "lanes": [{"width": LANE_WIDTH, "maxSpeed": LANE_MAX_SPEED}] * len(lane_setting[road]),
        "startIntersection": "intersection_%d_%d" % (x, y),
        "endIntersection": "intersection_%d_%d" % (x+moves[direction][0], y+moves[direction][1])
    }
    roads[road]["points"] = [
        intersections[roads[road]["startIntersection"]]["point"],
        intersections[roads[road]["endIntersection"]]["point"]
    ]

for road in lane_setting:
    _, x, y, direction = road.split('_')
    x = int(x)
    y = int(y)
    direction = int(direction)

    x += moves[direction][0]
    y += moves[direction][1]
    inter = "intersection_%d_%d" % (x, y)
    if intersections[inter]["virtual"]:
        continue
    for i, heads in enumerate(lane_setting[road]):
        for head in heads[1]:
            endRoad = "road_%d_%d_%d" % (x, y, (direction+turnmap[head]) % 4)
            laneLinks = []
            roadLink = {
                "type": typemap[head],
                "startRoad": road,
                "endRoad": endRoad,
                "laneLinks": laneLinks
            }
            for j in range(len(roads[endRoad]["lanes"])):
                laneLinks.append({
                    "startLaneIndex": i,
                    "endLaneIndex": j,
                    "points": findPath(roads[road], i, roads[endRoad], j, intersections[inter]["width"], NMIDPOINTS),
                })
            intersections[inter]["roadLinks"].append(roadLink)

# generate traffic light
for inter in intersections:
    if intersections[inter]["virtual"]:
        continue
    roadLinks = intersections[inter]["roadLinks"]
    roadLinkIndices = [x for x in range(len(roadLinks))]

    leftLaneLinks = set(filter(lambda x: roadLinks[x]["type"] == "turn_left", roadLinkIndices))
    rightLaneLinks = set(filter(lambda x: roadLinks[x]["type"] == "turn_right", roadLinkIndices))
    straightLaneLinks = set(filter(lambda x: roadLinks[x]["type"] == "go_straight", roadLinkIndices))

    WELaneLinks = set(filter(lambda x: roadLinks[x]["startRoad"].split('_')[3] == "0", roadLinkIndices))
    NSLaneLinks = set(filter(lambda x: roadLinks[x]["startRoad"].split('_')[3] == "3", roadLinkIndices))
    EWLaneLinks = set(filter(lambda x: roadLinks[x]["startRoad"].split('_')[3] == "2", roadLinkIndices))
    SNLaneLinks = set(filter(lambda x: roadLinks[x]["startRoad"].split('_')[3] == "1", roadLinkIndices))

    directionLaneLinks = {
        "N": NSLaneLinks,
        "S": SNLaneLinks,
        "W": WELaneLinks,
        "E": EWLaneLinks
    }
    turnLaneLinks = {
        "s": straightLaneLinks,
        "l": leftLaneLinks,
        "r": rightLaneLinks
    }

    tlPhases = []
    for i, phase in phase_setting[inter].items():
        if phase == "all red":
            tlPhases.append({
                "time": 5,
                "availableRoadLinks": rightLaneLinks
            })
        else:
            availableRoadLinks = set(rightLaneLinks)
            for phase_pair in phase:
                direction, turn = phase_pair
                availableRoadLinks |= directionLaneLinks[direction] & turnLaneLinks[turn]
            tlPhases.append({
                "time": 30,
                "availableRoadLinks": availableRoadLinks
            })
    for tlPhase in tlPhases:
        tlPhase["availableRoadLinks"] = list(tlPhase["availableRoadLinks"])
    intersections[inter]["trafficLight"]["lightphases"] = tlPhases

roadnet = {
    "intersections": [{"id": inter, **intersections[inter]} for inter in intersections],
    "roads": [{"id": road, **roads[road]} for road in roads]
}

json.dump(roadnet, open(path + "roadnet.json", "w"), indent=2)
            


            
        