from src.solvers.BASE_Solver import Solver
import copy
from time import process_time


class Baseline(Solver):

    def __init__(self, problem, method=None, seed=None):
        super().__init__(problem)
        self.method = method
        self.seed = seed

    def round_robin(self):
        start = process_time()
        schedule = {u['id']: [] for u in self.problem.S}
        for i in range(len(self.problem.P)):
            u_id = 1 + i % len(self.problem.U)
            schedule[u_id] += [self.problem.P[i]]
        stop = process_time()
        paths, makespan, conflicts_count = self.remove_conflicts(schedule)
        output = {
            'schedule': schedule,
            'paths': paths,
            'makespan': makespan,
            'makespan_wconflicts': self.problem.makespan(schedule),
            'conflicts_count': conflicts_count,
            'approximate': True,
            'runtime': stop - start
        }
        return output

    def JSIPM_best_approx(self):
        # list scheduling algorithm with the longest processing time rule is a 4/3-approximation algorithm for
        # scheduling jobs to minimize the makespan on identical parallel machines
        start = process_time()

        P = copy.deepcopy(self.problem.P)

        d = {}
        a = {}
        p = {}
        C = {}
        k = {}
        schedule = {}

        for u in self.problem.U:
            d[(u['id'], 0)] = u['start']
            a[(u['id'], 0)] = u['start']
            C[u['id']] = 0
            k[u['id']] = 0
            schedule[u['id']] = []

        while P != []:
            p_longest = max(P, key=lambda p: self.problem.DELTA[(p['src'], p['dst'])])
            u_firstIDLE = min(C, key=C.get)
            x = p_longest['src']
            y = p_longest['dst']
            z = a[(u_firstIDLE, k[u_firstIDLE])]

            tau_u_p = self.problem.DELTA[(z, x)] + self.problem.DELTA[
                (x, y)] + self.problem.delta_u + self.problem.delta_l

            k[u_firstIDLE] = k[u_firstIDLE] + 1
            C[u_firstIDLE] = tau_u_p + C[u_firstIDLE]
            d[(u_firstIDLE, k[u_firstIDLE])] = p_longest['src']  # departure
            a[(u_firstIDLE, k[u_firstIDLE])] = p_longest['dst']  # arrival
            p[(u_firstIDLE, k[u_firstIDLE])] = p_longest['id']
            schedule[u_firstIDLE] += [p_longest]
            P.remove(p_longest)

        stop = process_time()
        paths, makespan, conflicts_count = self.remove_conflicts(schedule)

        output = {
            'schedule': schedule,
            'paths': paths,
            'makespan': makespan,
            'makespan_wconflicts': self.problem.makespan(schedule),
            'conflicts_count': conflicts_count,
            'approximate': True,
            'runtime': stop - start
        }
        return output

    def solve(self):
        if self.method == "RR":
            return self.round_robin()
        else:
            assert self.method == "JSIPM"
            return self.JSIPM_best_approx()
