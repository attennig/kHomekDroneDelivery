"""
    This file contains the implementation of the branch-and-cut approach, namely UADP, proposed by
    Zhi Pei et al. in the paper (doi: 10.1109/TASE.2022.3184324):
        Urban On-Demand Delivery via Autonomous Aerial Mobility: Formulation and Exact Algorithm
    This implementation leverages Gurobi solver implementing the user-defined cuts in the callback function valid_inequalities_callback.
    At the moment the implementation consider only a subset of possible instances, indeed we ignore release times and deadlines for deliveries
    and the temporal horizon is set to a very large constant making it basically set to infinity.
"""
from src.solvers.BASE_Solver import Solver
from time import process_time
import random
import math
import gurobipy as gp
from gurobipy import GRB
from src.solvers.gurobi_param import OPT_TIME_LIMIT, OPT_MEM_LIMIT


def valid_inequalities_callback(model, where):
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            # Get the solution of the relaxed problem of the current node
            rel_landing_time_vars = model.cbGetNodeRel(model._landing_time_vars)
            rel_takeoff_time_vars = model.cbGetNodeRel(model._landing_time_vars)
            rel_arc_vars = model.cbGetNodeRel(model._arc_vars)

            # (22)
            for src in model._Pminus:
                lb = model._lb22[src]
                if rel_landing_time_vars[src] - lb < -0.1:
                    model.cbCut(
                        model._landing_time_vars[src] >= lb
                    )

            # (23)
            for dst in model._Pplus:
                lb = model._lb23[dst]
                if rel_landing_time_vars[dst] - lb < -0.1:
                    model.cbCut(
                        model._landing_time_vars[dst] >= lb
                    )

            # (24)
            for src in model._Pminus:
                lb = model._lb24[src]
                if rel_takeoff_time_vars[src] - lb < -0.1:
                    model.cbCut(
                        model._takeoff_time_vars[src] >= lb
                    )

            # (25)
            for dst in model._Pplus:
                lb = model._lb25[dst]
                if rel_takeoff_time_vars[dst] - lb < -0.1:
                    model.cbCut(
                        model._takeoff_time_vars[dst] >= lb
                    )

            # (26)
            val = len(model._Pminus) - len(model._Kminus) + sum([rel_arc_vars[(i, model._Kplus[0])] for i in model._Kminus])
            if sum([rel_arc_vars[(i, j)] for i in model._Pplus for j in model._Pminus]) - val < -0.1:
                model.cbCut(
                    gp.LinExpr([(1.0, model._arc_vars[(i, j)]) for i in model._Pplus for j in model._Pminus]) >= val)

            # (36) Capability inequality
            if sum([rel_arc_vars[(i, j)] for i in model._Kminus for j in model._Pminus]) < 1:
                model.cbCut(
                    gp.LinExpr(
                        [(1.0, model._arc_vars[(i, j)]) for i in model._Kminus for j in model._Pminus]
                    ) >= 1
                )

                # (30)
                S, obj = subtour_separation(model, rel_arc_vars)
                x_s = sum([rel_arc_vars[(i, j)] for i in model._Pplus for j in model._Pminus if i in S and j in S])
                n_min_drones = math.ceil(sum([model._costs[(i, i + len(model._Pminus))] for i in model._Pminus if i in S]) / model._H)
                if x_s > len(S) - n_min_drones:
                    model.cbCut(
                        gp.LinExpr(
                            [(1.0, model._arc_vars[(i, j)]) for i in model._Pplus for j in model._Pminus if i in S and j in S]
                        ) +
                        gp.LinExpr(
                            [(1.0, model._arc_vars[(i, i + len(model._Pminus))]) for i in model._Pminus if i in S]
                        )
                        >=
                        len(S) - n_min_drones  # math.ceil(sum([costs[(i,i+len(Pminus))] for i in Pminus if i in S])/H)
                    )

                    if obj <= 0:
                        # (38)
                        model.cbCut(
                            gp.LinExpr(
                                [(1.0, model._arc_vars[(i, j)]) for i in model._Kminus + model._Pplus for j in model._Pminus if i not in S if
                                 j in S]
                            ) >= 1
                        )


def subtour_separation(model, rel_arc_vars):
    s0 = []
    sBest = s0
    bestCandidate = s0
    tabuList = []
    tabuList.append(s0)
    maxTabuSize = 10
    max_itr = 25
    for i in range(max_itr):
        sNeighborhood = getNeighbors(model, bestCandidate)
        bestCandidateFitness = -GRB.INFINITY
        for sCandidate in sNeighborhood:
            if sCandidate not in tabuList and fitness(model, sCandidate, rel_arc_vars) > bestCandidateFitness:
                bestCandidate = sCandidate
                bestCandidateFitness = fitness(model, sCandidate, rel_arc_vars)

        if bestCandidateFitness > fitness(model, sBest, rel_arc_vars):
            sBest = bestCandidate

        tabuList.append(bestCandidate)
        if len(tabuList) > maxTabuSize:
            tabuList.pop(0)

    return sBest, fitness(model, sBest, rel_arc_vars)


def fitness(model, s, rel_arc_vars):
    # (37)
    assert s != None
    x_s = sum([rel_arc_vars[(i, j)] for i in model._Pplus for j in model._Pminus if i in s and j in s]) + \
          sum([rel_arc_vars[(i, i + len(model._Pminus))] for i in model._Pminus if i in s])
    min_n_drone = math.ceil(sum([model._costs[(i, i + len(model._Pminus))] for i in model._Pminus if i in s]) / model._H)
    return x_s + min_n_drone - len(s)


def getNeighbors(model, s):
    Neighbours = []
    # add
    if len(s) != len(model._Pminus):
        for p in model._Pminus:
            if p in s: continue
            neighbor = s.copy()
            neighbor.append(p)
            Neighbours.append(neighbor)
    # remove
    if len(s) != 0:
        for p in s:
            neighbor = s.copy()
            neighbor.remove(p)
            Neighbours.append(neighbor)
    return Neighbours


class UADP(Solver):

    def __init__(self, problem, only_greedy=False, args=None, horizon=10 ** 6):
        super().__init__(problem)
        self.args = args
        self.only_greedy = only_greedy

        self.horizon = horizon
        self.operation_time_in_channel = self.problem.delta_u + self.problem.delta_r / 2
        self.docks = {
            s['id']: []  # list of nodes that maps to station s
            for s in self.problem.S
        }

        # K set of drones
        self.K = len(self.problem.U)

        # P set of parcels
        self.deliveries = self.problem.P
        self.P = len(self.deliveries)

        if not only_greedy:
            self.nodes = []  # list(range(1,self.K+2*self.P+1))
            self.home_departures = set()
            self.home_arrivals = [] # it is only one in the paper implementation
            self.delivery_departures = set()
            self.delivery_arrivals = set()
            self.node_to_dock = {}
            self.available_time = {}
            self.delivery_to_nodes = {}

            for node_idx in range(1, self.K + 1):
                # drone home departure
                assert node_idx == self.problem.U[node_idx - 1]['id']
                self.docks[self.problem.U[node_idx - 1]['start']].append(node_idx)
                self.node_to_dock[node_idx] = self.problem.U[node_idx - 1]['start']
                self.nodes.append(node_idx)
                self.home_departures.add(node_idx)
                self.available_time[node_idx] = self.problem.U[node_idx - 1]['availability_time']

            for p_idx in range(1, self.P + 1):
                # pickup nodes
                src_node_idx = p_idx + self.K
                self.docks[self.deliveries[p_idx - 1]['src']].append(src_node_idx)
                self.node_to_dock[src_node_idx] = self.deliveries[p_idx - 1]['src']
                self.nodes.append(src_node_idx)
                self.delivery_departures.add(src_node_idx)
                # delivery nodes
                dst_node_idx = p_idx + self.K + self.P
                self.docks[self.deliveries[p_idx - 1]['dst']].append(dst_node_idx)
                self.node_to_dock[dst_node_idx] = self.deliveries[p_idx - 1]['dst']
                self.nodes.append(dst_node_idx)
                self.delivery_arrivals.add(dst_node_idx)
                self.delivery_to_nodes[self.deliveries[p_idx - 1]['id']] = (src_node_idx, dst_node_idx)

            # home return for every drone
            home_return_idx = self.K + 2 * self.P + 1
            home_return_dock = self.problem.U[0]['start']
            self.docks[home_return_dock].append(home_return_idx)
            self.node_to_dock[home_return_idx] = home_return_dock
            self.nodes.append(home_return_idx)
            self.home_arrivals.append(home_return_idx)

            # Preprocessing feasible arcs
            self.traverse_time = {}
            self.unused_arcs = set()
            for i in self.nodes:
                dock1 = self.node_to_dock[i]
                for j in self.nodes:
                    if i == j: continue
                    if j in self.home_departures and i != j: self.unused_arcs.add((i, j))
                    if i in self.home_arrivals and i != j: self.unused_arcs.add((i, j))
                    if i in self.delivery_departures and j != i + self.P: self.unused_arcs.add((i, j))
                    if j in self.delivery_arrivals and i != j - self.P: self.unused_arcs.add((i, j))
                    dock2 = self.node_to_dock[j]
                    if i in self.delivery_departures and j in self.delivery_arrivals:
                        p_idx = i - self.K - 1
                        self.traverse_time[(i, j)] = self.deliver(self.deliveries[p_idx])
                    else:
                        self.traverse_time[(i, j)] = self.delta(dock1, dock2)

            self.costs = self.traverse_time
            # Times to go back home
            for u in self.home_departures:
                dock1 = self.node_to_dock[u]
                for i in self.delivery_arrivals:
                    dock2 = self.node_to_dock[i]
                    self.costs[(i, u)] = self.delta(dock1, dock2)

    def solve(self):
        if self.only_greedy:
            schedule, makespan_wc, approx, runtime = self.initial_greedy_solution()
            paths, makespan, conflicts_count = self.remove_conflicts(schedule)
            output = {
                'schedule': schedule,
                'paths': paths,
                'makespan': makespan,
                'makespan_wconflicts': makespan_wc,
                'conflicts_count': conflicts_count,
                'approximate': approx,
                'runtime': runtime
            }
        else:
            model = self.setup()
            output = self.optimize(model)
        print(output)
        return output

    def get_arcs(self, schedule):
        arcs = []
        for u in self.problem.U:
            if len(schedule[u['id']]) == 0: continue
            nodes = self.delivery_to_nodes[schedule[u['id']][0]['id']]
            arcs.append((u['id'], nodes[0]))
            arcs.append((nodes[0], nodes[1]))
            for i in range(1, len(schedule[u['id']])):
                nodes = self.delivery_to_nodes[schedule[u['id']][i]['id']]
                arcs.append((arcs[-1][1], nodes[0]))
                arcs.append((nodes[0], nodes[1]))
            arcs.append((nodes[1], self.home_arrivals[0]))
        # print(arcs)
        return arcs

    def setup(self):
        env = gp.Env(empty=True)
        env.setParam('TimeLimit', OPT_TIME_LIMIT)
        env.setParam('SoftMemLimit', OPT_MEM_LIMIT)
        env.setParam('PreCrush', 1)
        env.start()
        model = gp.Model(env=env)
        # Find initial schedule to set start values for variables
        greedy_initial_schedule, _, _, _ = self.initial_greedy_solution()
        initial_arcs = self.get_arcs(greedy_initial_schedule)

        # Constants
        M = self.horizon + max(self.traverse_time.values())

        # Define variables
        makespan_var = model.addVar(vtype=GRB.CONTINUOUS, obj=1.0, name=f"C_max", lb=0, ub=self.horizon)
        arc_vars = model.addVars(self.traverse_time.keys(), vtype=GRB.BINARY, name='x')  # obj=self.traverse_time
        for (i, j) in initial_arcs:
            arc_vars[(i, j)].start = 1

        model.addConstrs((arc_vars[(i, j)] == 0 for (i, j) in self.unused_arcs), name="unused_arcs-")

        takeoff_time_vars = model.addVars(self.nodes, vtype=GRB.CONTINUOUS, name='d', lb=0, ub=GRB.INFINITY)

        landing_time_vars = model.addVars(self.nodes, vtype=GRB.CONTINUOUS, name='f', lb=0, ub=GRB.INFINITY)

        # Set objective function
        model.setObjective(makespan_var, GRB.MINIMIZE)

        # Add constraints
        # 11
        model.addConstrs((makespan_var >= takeoff_time_vars[i] for i in self.nodes if i not in self.home_arrivals),
                         name="11-")
        # 12
        for i in self.home_departures:
            model.addConstr(
                gp.LinExpr([(1.0, arc_vars[(i, j)]) for j in self.nodes if j in self.delivery_departures or j in self.home_arrivals]) == 1, name=f"12-{i}")
        # 13
        model.addConstr(arc_vars.sum('*', self.home_arrivals[0]) == self.K, name="13-")
        # 14
        model.addConstrs((arc_vars.sum(i, '*') == 0 for i in self.home_arrivals), name="14a-")
        model.addConstrs((arc_vars.sum(i, '*') == 1 for i in self.nodes if i not in self.home_arrivals), name="14b-")
        # 15
        model.addConstrs((arc_vars.sum('*', j) == 0 for j in self.home_departures), name="15a-")
        model.addConstrs((arc_vars.sum('*', j) == 1
                          for j in self.nodes if j not in self.home_arrivals and j not in self.home_departures),
                         name="15b-")
        # constraint 14a and 15a are useless with preprocessing
        # 16
        indices = [e for e in arc_vars.keys() if e[0] in self.delivery_departures and e[1] == e[0] + self.P]
        model.addConstrs((arc_vars[e] == 1 for e in indices), name="16-")
        # 17
        model.addConstrs((landing_time_vars[i] >= self.available_time[i] for i in self.home_departures), name="17-")
        # 18 useless in our case, we already set lb and ub
        # 19 x_ij = 1 --> d_i + T_ij == f_j
        # d_i + T_ij >= f_j - M(1-x_ij)
        # d_i + T_ij <= f_j + M(1-x_ij)
        indices = [e for e in arc_vars.keys() if e[1] not in self.home_arrivals]
        model.addConstrs(
            (takeoff_time_vars[e[0]] + self.traverse_time[e] >= landing_time_vars[e[1]] - M * (1 - arc_vars[e]) for e in
             indices), name="19a-")
        model.addConstrs(
            (takeoff_time_vars[e[0]] + self.traverse_time[e] <= landing_time_vars[e[1]] + M * (1 - arc_vars[e]) for e in
             indices), name="19b-")
        # 20
        model.addConstrs((landing_time_vars[i] <= takeoff_time_vars[i] for i in self.nodes), name="20-")
        # Spacial conflicts constraints
        # 1
        I_vars = model.addVars([n for n in self.nodes if n not in self.home_arrivals], vtype=GRB.BINARY, name='I')
        model.addConstrs((I_vars[i] == arc_vars[(i, self.home_arrivals[0])] for i in self.nodes if
                          (i, self.home_arrivals[0]) in arc_vars.keys()), name="1-")
        # 2
        indices_1 = [e for e in arc_vars.keys() if e[0] not in self.home_arrivals and e[1] not in self.home_arrivals]
        pi_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='pi')
        piu1_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='piu1')
        piu2_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='piu2')
        indices_1 = set(indices_1)
        model.addConstrs((pi_vars[(i, j)] >= arc_vars[(j, i)] for (i, j) in indices_1 if (j, i) in indices_1),
                         name="2b-")
        model.addConstrs((piu1_vars[e] + piu2_vars[e] >= 1 for e in indices_1), name="2c-")
        model.addConstrs((pi_vars[e] <= arc_vars[e] + M * (1 - piu1_vars[e]) for e in indices_1), name="2d-")
        model.addConstrs((pi_vars[(i, j)] <= arc_vars[(j, i)] + M * (1 - piu2_vars[(i, j)]) for (i, j) in indices_1 if
                          (j, i) in indices_1), name="2e-")
        # 3
        indices_2 = {
            dock: [(p, q) for p in self.docks[dock] for q in self.docks[dock]
                   if p not in self.home_departures and q not in self.home_departures and p != q]
            for dock in self.docks.keys()
        }
        for dock in self.docks.keys():
            if len(indices_2[dock]) == 0: continue
            pih_vars = model.addVars(indices_2[dock], vtype=GRB.CONTINUOUS, name='pih')
            piy1_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='piy1')
            piy2_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='piy2')
            model.addConstrs((pih_vars[(p, q)] >= self.operation_time_in_channel * (1 - pi_vars[(p, q)])
                              for (p, q) in indices_2[dock] if (p, q) in pi_vars.keys()),
                             name=f"3a{dock}-")
            model.addConstrs(
                (pih_vars[(p, q)] >= landing_time_vars[p] - landing_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"3b{dock}-")
            model.addConstrs(
                (pih_vars[(p, q)] >= - landing_time_vars[p] + landing_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"3c{dock}-")
            model.addConstrs(
                (pih_vars[(p, q)] <= landing_time_vars[p] - landing_time_vars[q] + M * (1 - piy1_vars[(p, q)]) for
                 (p, q) in indices_2[dock]),
                name=f"3d{dock}-"
            )
            model.addConstrs(
                (pih_vars[(p, q)] <= - landing_time_vars[p] + landing_time_vars[q] + M * (1 - piy2_vars[(p, q)]) for
                 (p, q) in
                 indices_2[dock]),
                name=f"3e{dock}-"
            )
            model.addConstrs((piy1_vars[(p, q)] + piy2_vars[(p, q)] >= 1 for (p, q) in indices_2[dock]),
                             name=f"3f{dock}-")
        # 4
        indices_1 = [e for e in arc_vars.keys() if e[0] not in self.home_arrivals and e[1] not in self.home_arrivals]
        ni_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='ni')
        niu1_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='niu1')
        niu2_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='niu2')
        niu3_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='niu3')
        niu4_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='niu4')
        indices_1 = set(indices_1)
        model.addConstrs((ni_vars[e] >= arc_vars[e] for e in indices_1), name="4a-")
        model.addConstrs((ni_vars[(i, j)] >= arc_vars[(j, i)] for (i, j) in indices_1 if (j, i) in indices_1),
                         name="4b-")
        model.addConstrs((ni_vars[e] >= I_vars[e[0]] for e in indices_1 if e[0] in I_vars.keys()), name="4c-")
        model.addConstrs((ni_vars[e] >= I_vars[e[1]] for e in indices_1 if e[1] in I_vars.keys()), name="4d-")

        model.addConstrs((niu1_vars[e] + niu2_vars[e] + niu3_vars[e] + niu4_vars[e] >= 1 for e in indices_1),
                         name="4e-")
        model.addConstrs((ni_vars[e] <= arc_vars[e] + M * (1 - niu1_vars[e]) for e in indices_1), name="4f-")
        model.addConstrs((ni_vars[e] <= arc_vars[(e[1], e[0])] + M * (1 - niu2_vars[e]) for e in indices_1 if
                          (e[1], e[0]) in indices_1), name="5g-")
        model.addConstrs(
            (ni_vars[e] <= I_vars[e[0]] + M * (1 - niu3_vars[e]) for e in indices_1 if e[0] in I_vars.keys()),
            name="5h-")
        model.addConstrs(
            (ni_vars[e] <= I_vars[e[1]] + M * (1 - niu4_vars[e]) for e in indices_1 if e[1] in I_vars.keys()),
            name="5i-")
        # 5
        indices_2 = {
            dock: [(p, q) for p in self.docks[dock] for q in self.docks[dock]]
            for dock in self.docks.keys()
        }
        for dock in self.docks.keys():
            nih_vars = model.addVars(indices_2[dock], vtype=GRB.CONTINUOUS, name='nih')
            niy1_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='niy1')
            niy2_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='niy2')
            model.addConstrs((nih_vars[(p, q)] >= self.operation_time_in_channel * (1 - ni_vars[(p, q)])
                              for (p, q) in indices_2[dock] if (p, q) in ni_vars.keys()),
                             name=f"5a{dock}-")
            model.addConstrs(
                (nih_vars[(p, q)] >= takeoff_time_vars[p] - takeoff_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"5b{dock}-")
            model.addConstrs(
                (nih_vars[(p, q)] >= - takeoff_time_vars[p] + takeoff_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"5c{dock}-")
            model.addConstrs(
                (nih_vars[(p, q)] <= takeoff_time_vars[p] - takeoff_time_vars[q] + M * (1 - niy1_vars[(p, q)]) for
                 (p, q) in indices_2[dock]),
                name=f"5d{dock}-"
            )
            model.addConstrs(
                (nih_vars[(p, q)] <= - takeoff_time_vars[p] + takeoff_time_vars[q] + M * (1 - niy2_vars[(p, q)]) for
                 (p, q) in
                 indices_2[dock]),
                name=f"5e{dock}-"
            )
            model.addConstrs((niy1_vars[(p, q)] + niy2_vars[(p, q)] >= 1 for (p, q) in indices_2[dock]),
                             name=f"5f{dock}-")
        # 6
        indices_1 = [e for e in arc_vars.keys() if e[0] not in self.home_arrivals and e[1] not in self.home_arrivals]
        fi_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='fi')
        fiu1_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='fiu1')
        fiu2_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='fiu2')
        fiu3_vars = model.addVars(indices_1, vtype=GRB.BINARY, name='fiu3')
        indices_1 = set(indices_1)
        model.addConstrs((fi_vars[e] >= arc_vars[e] for e in indices_1), name="6a-")
        model.addConstrs((fi_vars[(i, j)] >= arc_vars[(j, i)] for (i, j) in indices_1 if (j, i) in indices_1),
                         name="6b-")
        model.addConstrs((fi_vars[e] >= I_vars[e[1]] for e in indices_1 if e[1] in I_vars.keys()), name="6c-")

        model.addConstrs((fiu1_vars[e] + fiu2_vars[e] + fiu3_vars[e] >= 1 for e in indices_1), name="6d-")
        model.addConstrs((fi_vars[e] <= arc_vars[e] + M * (1 - fiu1_vars[e]) for e in indices_1), name="6e-")
        model.addConstrs((fi_vars[e] <= arc_vars[(e[1], e[0])] + M * (1 - fiu2_vars[e]) for e in indices_1 if
                          (e[1], e[0]) in indices_1), name="6f-")
        model.addConstrs((fi_vars[e] <= I_vars[e[1]] + M * (1 - fiu3_vars[e]) for e in indices_1 if
                          e[1] in I_vars.keys()), name="6g-")
        # 7
        indices_2 = {
            dock: [(p, q) for p in self.docks[dock] for q in self.docks[dock] if
                   p not in self.home_departures]
            for dock in self.docks.keys()
        }
        for dock in self.docks.keys():
            fih_vars = model.addVars(indices_2[dock], vtype=GRB.CONTINUOUS, name='fih')
            fiy1_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='fiy1')
            fiy2_vars = model.addVars(indices_2[dock], vtype=GRB.BINARY, name='fiy2')
            model.addConstrs(
                (fih_vars[(p, q)] >= self.operation_time_in_channel * (1 - fi_vars[(p, q)])
                 for (p, q) in indices_2[dock] if (p, q) in fi_vars.keys()),
                name=f"7a{dock}-")
            model.addConstrs(
                (fih_vars[(p, q)] >= landing_time_vars[p] - takeoff_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"7b{dock}-")
            model.addConstrs(
                (fih_vars[(p, q)] >= - landing_time_vars[p] + takeoff_time_vars[q] for (p, q) in indices_2[dock]),
                name=f"7c{dock}-")
            model.addConstrs(
                (fih_vars[(p, q)] <= landing_time_vars[p] - takeoff_time_vars[q] + M * (1 - fiy1_vars[(p, q)])
                 for (p, q) in indices_2[dock]),
                name=f"7d{dock}-"
            )
            model.addConstrs(
                (fih_vars[(p, q)] <= - landing_time_vars[p] + takeoff_time_vars[q] + M * (1 - fiy2_vars[(p, q)])
                 for (p, q) in indices_2[dock]),
                name=f"7e{dock}-"
            )
            model.addConstrs((fiy1_vars[(p, q)] + fiy2_vars[(p, q)] >= 1 for (p, q) in indices_2[dock]),
                             name=f"7f{dock}-")
        model.update()
        return model

    def optimize(self, model):
        # Store all needed values inside the model object
        model._vars = model.getVars()
        model._landing_time_vars = {
            int(var.VarName.split('[')[1].split(']')[0]): var
            for var in model.getVars() if var.VarName.split('[')[0] == 'f'
        }
        model._takeoff_time_vars = {
            int(var.VarName.split('[')[1].split(']')[0]): var
            for var in model.getVars() if var.VarName.split('[')[0] == 'd'
        }
        model._arc_vars = {
            (int(var.VarName.split('[')[1].split(']')[0].split(',')[0]),
             int(var.VarName.split('[')[1].split(']')[0].split(',')[1])): var
            for var in model.getVars() if var.VarName.split('[')[0] == 'x'
        }
        model._args_var = self.args
        model._H = self.horizon
        model._Kminus = self.home_departures
        model._Kplus = self.home_arrivals
        model._Pminus = self.delivery_departures
        model._Pplus = self.delivery_arrivals
        model._costs = self.costs
        model._lb22 = {
            src: min([model._costs[(i, j)] for (i, j) in model._costs.keys() if j == src]) for src in model._Pminus
        }
        model._lb23 = {
            dst: min(
                [model._costs[(k, dst - len(model._Pminus))] + model._costs[(dst - len(model._Pminus), dst)] for k in
                 model._Kminus]) for dst in model._Pplus
        }
        model._lb24 = {
            src: min([model._costs[(i, j)] for (i, j) in model._costs.keys() if j == src]) for src in model._Pminus
        }
        model._lb25 = {
            dst: min(
                [model._costs[(k, dst - len(model._Pminus))] + model._costs[(dst - len(model._Pminus), dst)] for k in
                 model._Kminus]) for dst in model._Pplus
        }
        model.optimize(valid_inequalities_callback)
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
            arc_values = {k:v.X for k, v in model._arc_vars.items()}
            takeoff_values = {k:v.X for k, v in model._takeoff_time_vars.items()}
            landing_values = {k:v.X for k, v in model._landing_time_vars.items()}
            schedule, paths, makespan = self.extract_solution(arc_values, takeoff_values, landing_values)
            print(f"time limit reached: {model.status == GRB.TIME_LIMIT}")
            print(f"optimal: {model.status == GRB.OPTIMAL}")
            output = {
                'schedule': schedule,
                'paths': paths,
                'makespan': makespan,
                'approximate': model.status != GRB.OPTIMAL,
                'runtime': model.Runtime
            }
            return output
        else:
            self.infeasible_debug(model)
            return None

    def extract_solution(self, arc_values, takeoff_values, landing_values):
        schedule, paths = {}, {}
        for u in self.problem.U:
            # print(f"Drone {u['id']}:")
            schedule[u['id']] = []
            paths[u['id']] = []

            home = u['id']
            current_node = self.get_next_node(home, arc_values)
            if self.node_to_dock[home] != self.node_to_dock[current_node] or current_node in self.home_arrivals:
                paths[u['id']].append({'station': self.node_to_dock[home],
                                       'landing_time': landing_values[home],
                                       'takeoff_time': takeoff_values[home],
                                       'available': landing_values[home],
                                       'delivery': None})
            while current_node != self.home_arrivals[0]:
                takeoff_time = takeoff_values[current_node]
                landing_time = landing_values[current_node]
                paths[u['id']].append({'station': self.node_to_dock[current_node],
                                       'landing_time': landing_time,
                                       'takeoff_time': takeoff_time,
                                       'available': landing_time,
                                       'delivery': None})
                if current_node in self.delivery_departures:
                    p_idx = current_node - self.K - 1
                    schedule[u['id']].append(self.deliveries[p_idx])
                    paths[u['id']][-1]['delivery'] = self.deliveries[p_idx]['id']
                current_node = self.get_next_node(current_node, arc_values)
            # print(f"\tEnds at its home station {u['home']}")
            paths[u['id']][-1]['takeoff_time'] = paths[u['id']][-1]['landing_time']
            if u['home'] != paths[u['id']][-1]['station']:
                landing = paths[u['id']][-1]['takeoff_time'] + self.delta(paths[u['id']][-1]['station'], u['home'])
                paths[u['id']].append({'station': u['home'],
                                       'landing_time': landing,
                                       'takeoff_time': None,
                                       'available': landing,
                                       'delivery': None})
            else:
                paths[u['id']][-1]['takeoff_time'] = None
        makespan = max(
            [paths[u['id']][-1]['landing_time'] for u in self.problem.U if paths[u['id']][-1]['landing_time']])
        return schedule, paths, makespan

    def get_next_node(self, current_node, arc_values):
        for (i, j) in arc_values.keys():
            if i == current_node and arc_values[(i, j)] >= 0.5:
                return j
        return None

    def infeasible_debug(self, model):
        assert model.status == GRB.INFEASIBLE
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

    def greedy_solution_time(self, drone, trip):
        time = 0
        last_pos = drone['start']
        for p in trip:
            time += self.delta(last_pos, p['src']) + self.deliver(p)
            last_pos = p['dst']
        return time

    def initial_greedy_solution(self):
        # Initial greedy solution:
        start = process_time()
        # P should be sorted with non-decreasing order time, but we have no order time
        last_pos = {}
        C = {}
        schedule = {}
        for u in self.problem.U:
            # Assign the start and end node to every drone as a null route
            schedule[u['id']] = []
            last_pos[u['id']] = u['start']
            C[u['id']] = 0
        for p in self.deliveries:
            min_loaded_drone = self.problem.U[0]  # k1
            nearest_drone = self.problem.U[0]  # k2
            # Get the currently minimum loaded drone k1 and nearest drone k2.
            for u in self.problem.U:
                if len(schedule[u['id']]) < len(schedule[min_loaded_drone['id']]):
                    min_loaded_drone = u
                if self.delta(last_pos[u['id']], p['src']) < self.delta(last_pos[nearest_drone['id']], p['src']):
                    nearest_drone = u
            # Calculate l1 = T(k1) + ti, k1 + δi, k1, l2 = T(k2) + ti, k2 + δi, k2
            time_min_loaded_drone = C[u['id']] + self.delta(last_pos[min_loaded_drone['id']], p['src']) + self.deliver(
                p)  # l1
            time_nearest_drone = C[u['id']] + self.delta(last_pos[nearest_drone['id']], p['src']) + self.deliver(
                p)  # l2
            if time_min_loaded_drone < time_nearest_drone:
                drone_to_assign = min_loaded_drone
            else:
                drone_to_assign = nearest_drone
            schedule[drone_to_assign['id']].append(p)
            C[drone_to_assign['id']] = C[drone_to_assign['id']] + self.delta(last_pos[drone_to_assign['id']],
                                                                             p['src']) + self.deliver(p)
            last_pos[drone_to_assign['id']] = p['dst']

        max_itr = 25
        for itr in range(max_itr):
            # intra-route swap operator
            u_id = max(C, key=C.get)
            best_makespan = C[u_id]
            best_swap = None
            for i in range(len(schedule[u_id])):
                for j in range(i + 1, len(schedule[u_id])):
                    if j <= i: continue
                    trip = schedule[u_id].copy()
                    trip[i], trip[j] = trip[j], trip[i]
                    new_makespan = self.greedy_solution_time(self.problem.get_drone_by_id(u_id), trip)
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_swap = (trip, new_makespan)
            if best_swap: schedule[u_id], C[u_id] = best_swap
            # Reinsertion operator
            u_id = max(C, key=C.get)
            best_makespan = C[u_id]
            p_j = schedule[u_id][random.randint(0, len(schedule[u_id]) - 1)]
            new_trip_uid = schedule[u_id].copy()
            new_trip_uid.remove(p_j)
            new_cost_uid = self.greedy_solution_time(self.problem.get_drone_by_id(u_id), new_trip_uid)
            best_reinsertion = None
            for u in self.problem.U:
                if u['id'] == u_id: continue
                for i in range(len(schedule[u['id']]) + 1):
                    trip_u = schedule[u['id']].copy()
                    trip_u.insert(i, p_j)
                    new_cost_u = self.greedy_solution_time(u, trip_u)
                    new_makespan = max(
                        [new_cost_u, new_cost_uid] + [C[k] for k in C.keys() if k != u_id and k != u['id']])
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_reinsertion = (u, trip_u, new_cost_u)
            if best_reinsertion:
                schedule[best_reinsertion[0]['id']] = best_reinsertion[1]
                C[best_reinsertion[0]['id']] = best_reinsertion[2]
                schedule[u_id] = new_trip_uid
                C[u_id] = new_cost_uid
            # Intra-route swap operator
            u_r, u_s = random.sample(self.problem.U, 2)
            if len(schedule[u_r['id']]) == 0 or len(schedule[u_s['id']]) == 0: continue
            i = random.randint(0, len(schedule[u_r['id']]) - 1)
            j = random.randint(0, len(schedule[u_s['id']]) - 1)
            p_i = schedule[u_r['id']][i]
            schedule[u_r['id']][i] = schedule[u_s['id']][j]
            schedule[u_s['id']][j] = p_i
            C[u_r['id']] = self.greedy_solution_time(u_r, schedule[u_r['id']])
            C[u_s['id']] = self.greedy_solution_time(u_s, schedule[u_s['id']])
        stop = process_time()
        return schedule, max(C.values()), True, stop - start
