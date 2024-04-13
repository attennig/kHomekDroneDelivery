import networkx as nx
import random
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
from src.DroneModel import DroneModel
import json

# Environmental parameters
air_speed = 10  # m/s
drag_coefficient = 0.66
drone_front_area = 1  # m^2
air_density = 1.293  # kg/m^3
drone_weight = 44  # kg
drone_width = 3  # m
B = 16416000  # Joule

AoI_side = 30000  # m
max_dist = 8000  # m


def ground_speed(air_speed, wind_speed, wind_direction, station_i, station_j):
    # Computes the ground speed of the drone flying from station_i to station_j (in m/s) considering the wind speed
    # and direction

    # Compute angle between the stations
    x_i, y_i, x_j, y_j = station_i[0], station_i[1], station_j[0], station_j[1]
    if x_i == x_j:
        if y_i < y_j:
            angle_ij = np.pi / 2
        else:
            angle_ij = -np.pi / 2
    elif y_i == y_j:
        if x_i < x_j:
            angle_ij = 0
        else:
            angle_ij = np.pi
    else:
        angle_ij = np.arctan2(y_j - y_i, x_j - x_i)

    # We assume the wind speed is always lower than the air speed
    assert wind_speed < air_speed

    # Compute ground speed
    minus_half_b = wind_speed * (np.cos(angle_ij) * np.cos(wind_direction) + np.sin(angle_ij) * np.sin(wind_direction))
    ground_speed_ij = minus_half_b + np.sqrt(minus_half_b ** 2 + air_speed ** 2 - wind_speed ** 2)
    return ground_speed_ij


wind_directions = np.array([90, 60, 30, 0, 330, 300, 270, 240, 210, 180, 150, 120])
wind_directions = list(np.radians(wind_directions))
wind_stats = [0.12, 0.17, 0.08, 0.05, 0.07, 0.14, 0.09, 0.09, 0.09, 0.05, 0.02, 0.03]

drone = DroneModel("data/drone_models/octocopter.json", airspeed=air_speed)


def get_random_wind_directions(n=10):
    return random.choices(wind_directions, k=n, weights=wind_stats)


def station_distance(station_i, station_j):
    return np.sqrt((station_j[0] - station_i[0]) ** 2 + (station_j[1] - station_i[1]) ** 2)


def is_wind_feasible(G, pos, wind_direction, wind_speed):
    W = nx.DiGraph()
    W.add_nodes_from(G.nodes)
    for u, v in G.edges():
        if ground_speed(air_speed, wind_speed, wind_direction, pos[u], pos[v]) * drone.max_flight_time(6.5) \
                > station_distance(pos[u], pos[v]):
            W.add_edge(u, v)
    return nx.is_strongly_connected(W)


def generate_instance(path, n_s):
    success = 0
    tries = 0
    while success < 1:
        seed = random.randint(0, 1000000000000000000000)
        G = nx.generators.random_geometric_graph(n_s, 1 / AoI_side * max_dist, seed=seed)
        if nx.is_connected(G):
            G = G.to_directed()
            pos = nx.get_node_attributes(G, "pos")
            for node in pos.keys():
                pos[node] = [pos[node][0] * AoI_side, pos[node][1] * AoI_side]

            # Generate winds for the instance
            winds = []
            while len(winds) < 20:
                wind_direction = get_random_wind_directions(1)[0]
                wind_speed = random.triangular(0, 13.33, 4.92)
                if is_wind_feasible(G, pos, wind_direction, wind_speed):
                    winds.append({"id": len(winds) + 1, "direction": wind_direction, "speed": wind_speed})
            success += 1

            # Save instance map and original graph
            nx.draw(G, pos=pos, with_labels=True)
            plt.savefig(f"{path}/map.png")
            plt.savefig(f"{path}/../maps/map_{seed}.png")
            plt.close()
            nx.write_gml(G, f"{path}/graph.gpickle")
            json.dump(winds, open(f"{path}/winds.json", "w"))
            with open(f"{path}/seed.txt", "w") as f:
                f.write(str(seed))
            print(f"Generated instance {path}")

            # Save station positions
            facilities = []
            for node in pos.keys():
                facilities.append({
                    "id": node + 1,
                    "x": pos[node][0],
                    "y": pos[node][1]
                })
            json.dump(facilities, open(f"{path}/facilities.json", "w"))

            # Generate demand
            demands = []
            while len(demands) < n_s * 10:
                demand = {
                    "id": len(demands) + 1,
                    "src": random.randint(1, n_s),
                    "dst": random.randint(1, n_s),
                    "weight": random.uniform(0.5, 6.5)
                }
                if demand["src"] != demand["dst"]:
                    demands.append(demand)
            json.dump(demands, open(f"{path}/demand.json", "w"))

            # Generate fleet
            fleet = []
            while len(fleet) < n_s:
                fleet.append({
                    "id": len(fleet) + 1,
                    "home": len(fleet) + 1,
                })
            json.dump(fleet, open(f"{path}/fleet.json", "w"))

        tries += 1


if not os.path.exists("data/experiments"):
    os.mkdir("data/experiments")

n_s = int(sys.argv[1])
n_e = int(sys.argv[2])

if not os.path.exists(f"data/experiments/n_{n_s}"):
    os.mkdir(f"data/experiments/n_{n_s}")
if not os.path.exists(f"data/experiments/n_{n_s}/maps"):
    os.mkdir(f"data/experiments/n_{n_s}/maps")
path = f"data/experiments/n_{n_s}/e_{n_e}"
if not os.path.exists(path):
    os.mkdir(path)
generate_instance(path, n_s)
