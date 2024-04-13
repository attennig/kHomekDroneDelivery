import math
import networkx as nx
import json
from src.DroneModel import DroneModel
import numpy as np

class ProblemInstance:
    def __init__(self, stations, drones, deliveries, param_file="data/parameters/default.json", wind_speed=None,
                 wind_direction=None, drone_model="data/drone_models/octocopter.json"):
        self.S = stations
        self.U = drones
        self.P = deliveries

        self.P_byid = {p['id']: p for p in self.P}
        self.U_byid = {u['id']: u for u in self.U}
        self.S_byid = {s['id']: s for s in self.S}

        with open(param_file) as f:
            param = json.load(f)

        self.delta_u = float(param["delta_u"])  # time to unload a package
        self.delta_l = float(param["delta_l"])  # time to load a package
        self.delta_r = float(param["delta_r"])  # time to recharge (or battery swap) a drone
        self.air_speed = float(param["drone_speed"])  # Target air speed of the drone

        if wind_speed and wind_direction:
            self.wind_speed = wind_speed
            self.wind_direction = wind_direction
        else:
            self.wind_speed = 0.0
            self.wind_direction = 0.0

        self.drone = DroneModel(drone_model, airspeed=self.air_speed)

        # Shortest paths between all stations, distance between all stations (in terms of time) and energy consumed
        self.DELTA, self.LCP, self.ENERGY = dict(), dict(), dict()
        self.DELTA_weighted, self.LCP_weighted, self.ENERGY_weighted = dict(), dict(), dict()
        self.precompute_costs()

    def __repr__(self):
        return f"ProblemInstance:\n- Stations:{self.S},\n- Drones:{self.U},\n- Deliveries{self.P}"

    @staticmethod
    def lcp(G, x, y):
        try:
            path = nx.shortest_path(G, source=x, target=y, weight='weight')
            cost = nx.path_weight(G, path, "weight")
            energy_cost = nx.path_weight(G, path, "energy_consumption")
            return path, cost, energy_cost
        except nx.NetworkXNoPath:
            return [], 0, 0

    def ground_speed(self, station_i, station_j):
        # Computes the ground speed of the drone flying from station_i to station_j (in m/s) considering the wind speed
        # and direction

        # Compute angle between the stations
        x_i, y_i, x_j, y_j = station_i['x'], station_i['y'], station_j['x'], station_j['y']
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

        # Compute ground speed
        minus_half_b = self.wind_speed * (
                    np.cos(angle_ij) * np.cos(self.wind_direction) + np.sin(angle_ij) * np.sin(self.wind_direction))
        ground_speed_ij = minus_half_b + np.sqrt(minus_half_b ** 2 + self.air_speed ** 2 - self.wind_speed ** 2)
        return ground_speed_ij

    def flight_time(self, station_i, station_j):
        # Computes the time needed to travel from station_i to station_j
        distance_ij = math.sqrt((station_j['x'] - station_i['x']) ** 2 + (station_j['y'] - station_i['y']) ** 2)
        ground_speed_ij = self.ground_speed(station_i, station_j)
        delta_ij = distance_ij / ground_speed_ij
        return delta_ij

    def compute_edges(self, set1: list, set2: list, w=.0):
        # Compute edges between two sets of stations
        # set1 and set2 are lists of dictionaries representing stations
        # w is the weight of the delivery
        edges = []
        for x in set1:
            for y in set2:
                x_id, y_id = x["id"], y["id"]
                if x_id == y_id:
                    continue
                flight_time = self.flight_time(x, y)
                if flight_time > self.drone.max_flight_time(w):
                    continue
                time = flight_time + self.delta_r
                energy_consumption = flight_time * self.drone.power_consumption(w)
                edges += [(x_id, y_id, {"weight": time, "energy_consumption": energy_consumption})]
        return edges

    def transport_graph(self, w=.0, additional_stations=None):
        transport_graph = nx.DiGraph()
        transport_graph.add_nodes_from([s['id'] for s in self.S])
        if additional_stations:
            transport_graph.add_nodes_from([s['id'] for s in additional_stations])
        transport_graph.add_edges_from(self.compute_edges(self.S, self.S, w))
        return transport_graph

    def compute_shortest_path(self, fictitious_station, station, w=.0, graph=None):
        if not graph:
            graph = self.transport_graph(w, additional_stations=[fictitious_station])
        path, cost, energy = self.lcp(graph, fictitious_station['id'], station)
        return path, cost - self.delta_r, energy

    def precompute_costs(self):
        self.precompute_delivery_costs()
        self.precompute_movement_costs()

    def precompute_delivery_costs(self):
        G_T = {}
        max_weight = int(np.ceil(max([p['weight'] for p in self.P])))
        for w in range(0, max_weight+1):
            G_T[w] = self.transport_graph(w)

        for p in self.P:
            w = int(np.ceil(p['weight']))
            self.LCP_weighted[p['id']], self.DELTA_weighted[p['id']], self.ENERGY_weighted[p['id']] = self.lcp(G_T[w], p['src'], p['dst'])
            self.DELTA_weighted[p['id']] += self.delta_l + self.delta_u

    def precompute_movement_costs(self):
        G_T = self.transport_graph()
        endpoints = set()
        for p in self.P:
            endpoints.add(p['src'])
            endpoints.add(p['dst'])
        for u in self.U:
            endpoints.add(u['start'])
            endpoints.add(u['home'])
        for u in endpoints:
            for v in endpoints:
                self.LCP[(u, v)], self.DELTA[(u, v)], self.ENERGY[(u, v)] = self.lcp(G_T, u, v)

    def get_drone_by_id(self, id):
        return self.U_byid[id]

    def completion_time(self, u, trip):
        time = u['availability_time']
        if (not trip) or (trip == [[]]):
            return time + self.DELTA[(u['start'], u['home'])]
        time += self.DELTA[(u['start'], trip[0]['src'])]
        time += self.DELTA_weighted[trip[0]['id']]
        for i in range(1, len(trip)):
            p = trip[i]
            q = trip[i - 1]
            time += self.DELTA[(q['dst'], p['src'])]
            time += self.DELTA_weighted[p['id']]
        time += self.DELTA[(trip[-1]['dst'], u['home'])]
        return time

    def makespan(self, schedule):
        times = []
        for u in self.U:
            times.append(self.completion_time(u, schedule[u['id']]))
        return max(times)
