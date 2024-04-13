from src.solvers.BASE_Solver import Solver
import gurobipy as gp
from gurobipy import GRB
from src.solvers.gurobi_param import OPT_TIME_LIMIT, OPT_MEM_LIMIT

class MILP(Solver):
    def __init__(self, problem, args):
        super().__init__(problem)
        self.args = args

        # Gurobi solver
        env = gp.Env(empty=True)
        if args.algorithm == "MILP":
            env.setParam('TimeLimit', OPT_TIME_LIMIT)
        env.setParam('SoftMemLimit', OPT_MEM_LIMIT)
        env.start()
        self.model = gp.Model(env=env)
        self.setup()

    def solve(self):
        self.model.optimize()
        if self.model.SolCount > 0:
            schedule, makespan, approximate = self.extract_solution(self.model.Status)
            paths, new_makespan, conflicts_count = self.remove_conflicts(schedule)
            output = {
                'schedule': schedule,
                'paths': paths,
                'makespan': new_makespan,
                'makespan_wconflicts': makespan,
                'conflicts_count': conflicts_count,
                'approximate': approximate,
                'runtime': self.model.Runtime
            }
            return output
        else:
            return None

    def setup(self):
        # Indices
        P_ids = [_p["id"] for _p in self.problem.P]  #: indices of deliveries
        U_ids = [_u["id"] for _u in self.problem.U]  #: indices of drones

        # Constants
        arrival = lambda i: [_p["dst"] for _p in self.problem.P if _p["id"] == i][0] if i <= len(P_ids) else \
        [_u['start'] for _u in self.problem.U if _u["id"] == i - len(P_ids)][0]
        departure = lambda i: [_p["src"] for _p in self.problem.P if _p["id"] == i][0] if i <= len(P_ids) else \
        [_u['start'] for _u in self.problem.U if _u["id"] == i - len(P_ids)][0]

        c = lambda i, j: self.problem.DELTA[(arrival(i), departure(j))] + self.problem.DELTA[(departure(j), arrival(j))]

        # Variables
        x = {}  # x_{u},{i},{j}
        z = {}  # z_{u},{i}

        for u in U_ids:
            for i in P_ids + [len(P_ids) + u]:
                z[f"{u},{i}"] = self.model.addVar(vtype=GRB.INTEGER, name=f"z_{u},{i}", ub=len(P_ids), lb=0)
                for j in P_ids + [len(P_ids) + u]:
                    if i == j: continue
                    x[f"{u},{i},{j}"] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{u},{i},{j}")

        maxC = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"maxC", lb=0)

        # Constraints

        #  Define objective function
        #  \gamma \ge \sum_{i,j \in P \cup \{h(u)\}} \chi_u(i,j) \cdot c(i,j) \quad \forall u \in U\label{eq:obj1}
        for u in U_ids:
            self.model.addConstr(
                maxC >= gp.LinExpr(
                    [(c(i, j), x[f"{u},{i},{j}"]) for i in P_ids + [len(P_ids) + u] for j in P_ids + [len(P_ids) + u] if
                     i != j]
                ),
                f"obj1_{u}")

        #  Each tour starts from home stations
        #  \sum_{i \in P {\color{red} \cup \{h(u)\}}} \chi_u(h(u),i) = 1, \quad \forall u \in U\label{eq:min1}
        for u in U_ids:
            self.model.addConstr(
                1 == gp.LinExpr(
                    [(1.0, x[f"{u},{len(P_ids) + u},{i}"]) for i in P_ids + [len(P_ids) + u] if i != len(P_ids) + u]
                ),
                f"min1_{u}")

        #  Each delivery is in exactly one tour
        #  \sum_{j \in P  \cup \{h(u)\}} \chi_u(i,j) = 1, \quad \forall i \in P\label{eq:min2}
        for i in P_ids:
            self.model.addConstr(
                1 == gp.LinExpr(
                    [(1.0, x[f"{u},{i},{j}"]) for u in U_ids for j in P_ids + [len(P_ids) + u] if i != j]
                ),
                f"min2_{i}")

        #  Flow constraints
        #  \sum_{j \in P  \cup \{h(u)\}} \chi_u(j,i) = \sum_{j \in P  \cup \{h(u)\}} \chi_u(i,j), \quad \forall u \in U,\forall i \in P \cup \{h(u)\} \label{eq:min3}
        for u in U_ids:
            for i in P_ids + [len(P_ids) + u]:
                self.model.addConstr(
                    gp.LinExpr(
                        [(1.0, x[f"{u},{j},{i}"]) for j in P_ids + [len(P_ids) + u] if i != j]
                    )
                    == gp.LinExpr(
                        [(1.0, x[f"{u},{i},{j}"]) for j in P_ids + [len(P_ids) + u] if i != j]
                    ),
                    f"min3_{u},{i}")

        #  Home in sequence
        #  z_u(h(u)) = 0, \quad \forall u \in U\label{eq:min4}
        for u in U_ids:
            self.model.addConstr(
                z[f"{u},{len(P_ids) + u}"] == 0,
                f"min4_{u}")

        #  sequence numbering in monotonically increasing
        #  z_u(j) \ge z_u(i) + (\chi_u(i,j) - |P|(1-\chi_u(i,j))), \quad \forall u \in U, i \in P \cup \{h(u)\}, j \in P\label{eq:min5}
        for u in U_ids:
            for i in P_ids + [len(P_ids) + u]:
                for j in P_ids:
                    if i == j: continue
                    self.model.addConstr(
                        z[f"{u},{j}"] >= z[f"{u},{i}"] + (x[f"{u},{i},{j}"] - len(P_ids) * (1 - x[f"{u},{i},{j}"])),
                        f"min5_{u}")

        # Objective function
        sum_positions = gp.LinExpr([(1.0, z[f"{u},{i}"]) for u in U_ids for i in P_ids])
        self.model.setObjective(maxC, GRB.MINIMIZE)
        self.model.write(f"data/log/model_v2.lp")

    def extract_solution(self, status_code):
        approximate = False
        if status_code in [GRB.TIME_LIMIT, GRB.MEM_LIMIT]:
            approximate = True

        schedule = {}
        for u in self.problem.U:
            T_u = []
            home_id = u["id"] + len(self.problem.P)
            next_delivery = {'id': home_id}
            while True:
                next_deliveries = [j for j in self.problem.P + [{'id': home_id}] if j['id'] != next_delivery['id'] if
                                 self.model.getVarByName(f"x_{u['id']},{next_delivery['id']},{j['id']}").x >= 0.5]
                if next_deliveries: next_delivery = next_deliveries[0]
                if next_delivery["id"] == home_id: break
                T_u += [next_delivery]
            schedule[u['id']] = T_u

        return schedule, self.model.getVarByName(f"maxC").x, approximate

    def print_vars(self):
        print("VARS ______")
        for var in self.model.getVars():
            print(f"{var.VarName} = {var.x}")
        print("VARS ______ END")
