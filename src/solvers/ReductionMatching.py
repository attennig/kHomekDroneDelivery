import numpy as np
from src.solvers.BASE_Solver import Solver
from src.solvers.kCRANES import kCranesSolver
import networkx as nx
import random
from time import process_time


def centroid(stations_x, stations_y):
    length = len(stations_x)
    sum_x = sum(stations_x)
    sum_y = sum(stations_y)
    return sum_x / length, sum_y / length


def check_feasibility(edges, costs, c, c_0, c_1, base_graph, top_nodes):
    G = base_graph.copy()
    G.add_weighted_edges_from(edges[: costs.searchsorted(c, side="right") + 1])
    matching = nx.bipartite.maximum_matching(G, top_nodes)
    if nx.is_perfect_matching(G, matching):
        c_1 = c
    else:
        c_0 = c
    return c_0, c_1


class ReductionMatching(Solver):
    def __init__(self, problem, args=None):
        super().__init__(problem)
        self.args = args

    def get_arc_delivery_id(self, e):
        return int(e[0][3:])

    def constructCSPinstance(self, initial_vertex):
        V = []
        A = []
        for p in self.problem.P:
            V += [(f"src{p['id']}", {"id_station": p['src']}), (f"dst{p['id']}", {'id_station': p['dst']})]
            A += [(f"src{p['id']}", f"dst{p['id']}", {'weight': self.deliver(p)})]
        E = [(u[0], v[0], {'weight': self.delta(u[1]['id_station'], v[1]['id_station'])}) for u in V for v in V if
             v != u]
        V += ['init']
        t_graph_augmented = self.problem.transport_graph(0, additional_stations=[initial_vertex])
        E += [('init', v[0], {'weight': self.delta_from_arbitrary_point(initial_vertex, v[1]['id_station'], graph=t_graph_augmented)}) for v in V
              if v != 'init']
        return V, E, A

    def random_assignment(self, tours):
        assignment = random.sample(range(1, len(tours) + 1), len(tours))
        assignment = {u['id']: tours[assignment[u['id'] - 1]] for u in self.problem.U}
        return assignment

    def greedy_assignment(self, T):
        tours = T.copy()
        assignment = {}
        drones = self.problem.U.copy()
        while tours:
            u, tour = None, None
            min_assignment_cost = float('inf')
            for u in drones:
                for i, t in enumerate(tours):
                    if self.args.simulation == 'static':
                        cost = self.problem.DELTA[(u['start'], t[0][0])] + self.problem.DELTA[(t[-1][1], u['start'])]
                    else:
                        assert self.args.simulation == 'dynamic'
                        cost = u['time_to_complete_running_task'] + self.problem.DELTA[
                            (u['start'], t[0][0])] + self.problem.DELTA[(t[-1][1], u['start'])]
                    if cost < min_assignment_cost:
                        min_assignment_cost = cost
                        u, tour = u, t
            assignment[u['id']] = tour
            tours.remove(tour)
            drones.remove(u)
        return assignment

    def bottleneck_matching(self, edgelist, n):
        base_graph = nx.Graph()
        top_nodes = {float(i + 1) for i in range(n)}
        base_graph.add_nodes_from(top_nodes, bipartite=0)
        base_graph.add_nodes_from([float(i + 1 + n) for i in range(n)], bipartite=1)
        edges = np.array(sorted(edgelist, key=lambda x: x[2]))
        costs = edges[:, 2]
        c_0 = np.min(costs)
        c_1 = np.max(costs)
        if c_0 == c_1:
            # Any permutation is optimal, we return the identity permutation
            return {i + 1: i + 1 + n for i in range(n)}
        else:
            c_star = costs[costs.searchsorted(c_0, side="right"):costs.searchsorted(c_1, side="left")]
            c = None
            while c_star.size != 0:
                c = c_star[np.ceil(c_star.size / 2).astype(int) - 1]
                c_0, c_1 = check_feasibility(edges, costs, c, c_0, c_1, base_graph, top_nodes)
                c_star = costs[costs.searchsorted(c_0, side="right"):costs.searchsorted(c_1, side="left")]
            if c != c_0:
                c_0, c_1 = check_feasibility(edges, costs, c_0, c_0, c_1, base_graph, top_nodes)
            G = base_graph.copy()
            G.add_weighted_edges_from(edges[: costs.searchsorted(c_1, side="right") + 1])
            matching = nx.bipartite.minimum_weight_full_matching(G, top_nodes)
            assert nx.is_perfect_matching(G, matching)
            return matching

    def bottleneck_matching_assignment(self, tours):
        n = len(self.problem.U)
        edgelist = []
        for j, tour in tours.items():
            j += n
            if not tour:
                for u in self.problem.U:
                    edgelist.append([u['id'], j, 0])
                continue
            start = self.problem.P_byid[int(tour[0][1][3:])]['src']
            end = self.problem.P_byid[int(tour[-1][0][3:])]['dst']
            cost = 0
            for m in tour:
                if len(m) == 3:
                    cost += m[2]['weight']
                if len(m) == 2 and m[0] != 'init' and m[1] != 'init':
                    src = self.problem.P_byid[int(m[0][3:])][m[0][:3]]
                    dst = self.problem.P_byid[int(m[1][3:])][m[1][:3]]
                    if src != dst:
                        cost += self.delta(src, dst)
            for u in self.problem.U:
                edgelist.append([u['id'],
                                 j,
                                 u['time_to_complete_running_task'] + self.delta(u['start'], start) + cost + self.delta(end, u['home'])])
        matching = self.bottleneck_matching(edgelist, n)
        assignment = {u['id']: tours[matching[u['id']] - n] for u in self.problem.U}
        return assignment

    def get_schedule_and_cost(self, assignment, A):
        schedule = {}
        costs = {}
        # extract delivery list from arcs in tours
        for u in self.problem.U:
            deliveries = []
            for e in assignment[u['id']]:
                if e in A:
                    deliveries.append(self.problem.P_byid[self.get_arc_delivery_id(e)])

            schedule[u['id']] = deliveries
            costs[u['id']] = 0
            if deliveries:
                if self.args.simulation == 'static':
                    costs[u['id']] += self.delta(u['start'], deliveries[0]['src']) + self.deliver(
                        deliveries[0])
                else:
                    assert self.args.simulation == 'dynamic'
                    costs[u['id']] += u['time_to_complete_running_task'] + self.delta(u['start'],
                                                                                      deliveries[0][
                                                                                          'src']) + self.deliver(
                        deliveries[0])
                for idx in range(1, len(deliveries)):
                    p = deliveries[idx]
                    q = deliveries[idx - 1]
                    costs[u['id']] += self.delta(q['dst'], p['src'])
                    costs[u['id']] += self.deliver(p)

                costs[u['id']] += self.delta(deliveries[-1]['dst'], u['home'])
        return schedule, costs

    def solve(self):
        approximate = True
        if not self.problem.P: return {u['id']: [] for u in self.problem.U}, 0, False
        # h is the centroid of the home stations
        h = {'id': len(self.problem.S) + 1}
        home_stations = [s for s in self.problem.S if s['id'] in [u['start'] for u in self.problem.U]]
        h['x'], h['y'] = centroid([s['x'] for s in home_stations],
                                  [s['y'] for s in home_stations])
        # construct CSP instance
        V, E, A = self.constructCSPinstance(h)
        # solve kCRANES
        kCRANES_solver = kCranesSolver(V, 'init', E, A, len(self.problem.U))
        start = process_time()
        tours = kCRANES_solver.kCRANES()
        # bottleneck matching
        tour_assignment = self.bottleneck_matching_assignment(tours)
        stop = process_time()
        schedule, costs = self.get_schedule_and_cost(tour_assignment, A)
        makespan = costs[max(costs, key=costs.get)]
        paths, new_makespan, conflicts_count = self.remove_conflicts(schedule)
        output = {
            'schedule': schedule,
            'paths': paths,
            'makespan': new_makespan,
            'makespan_wconflicts': makespan,
            'conflicts_count': conflicts_count,
            'approximate': approximate,
            'runtime': stop - start
        }
        return output
