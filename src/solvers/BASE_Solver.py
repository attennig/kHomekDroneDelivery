import itertools
import numpy as np
from src.Problem import ProblemInstance
import networkx as nx


class Solver:

    def __init__(self, problem_instance: ProblemInstance):
        self.problem = problem_instance

    def solve(self):
        pass

    def delta(self, x, y):
        # x,y are stations
        return self.problem.DELTA[(x, y)]

    def delta_from_arbitrary_point(self, pnt, x, graph=None):
        _, cost, _ = self.problem.compute_shortest_path(pnt, x, graph=graph)
        return cost

    def deliver(self, p: dict):
        return self.problem.DELTA_weighted[p['id']]

    def get_paths(self, schedule):
        path = {
            u['id']: []
            for u in self.problem.U
        }
        for u in self.problem.U:
            time = u['availability_time']
            path[u['id']].append(
                {'station': u['start'], 'landing_time': time, 'takeoff_time': None, 'available': time, 'delivery': None}
            )
            for p in schedule[u['id']]:
                last_station_idx = path[u['id']][-1]['station']
                # time += self.problem.delta_r
                path[u['id']][-1]['takeoff_time'] = time
                if len(self.problem.LCP[(last_station_idx, p['src'])]) > 1:
                    time += self.delta(last_station_idx, p['src'])
                    path[u['id']].append({'station': p['src'],
                                          'landing_time': time - self.problem.delta_r,
                                          'takeoff_time': time + self.problem.delta_l,
                                          'available': time,
                                          'delivery': p['id']})
                else:
                    path[u['id']][-1]['takeoff_time'] += self.problem.delta_l
                    path[u['id']][-1]['delivery'] = p['id']
                time += self.deliver(p)
                path[u['id']].append({'station': p['dst'],
                                      'landing_time': time - self.problem.delta_u - self.problem.delta_r,
                                      'takeoff_time': None,
                                      'available': time,
                                      'delivery': None})
            if len(self.problem.LCP[(path[u['id']][-1]['station'], u['home'])]) > 1:
                # time += self.problem.delta_r
                path[u['id']][-1]['takeoff_time'] = time
                time += self.delta(path[u['id']][-1]['station'], u['home'])
                path[u['id']].append(
                    {'station': u['home'],
                     'landing_time': time - self.problem.delta_r,
                     'available': time,
                     'takeoff_time': None,
                     'delivery': None})
        times = {
            u['id']: path[u['id']][-1]['available']
            for u in self.problem.U
        }
        return path, times

    @staticmethod
    def next_longest_earliest_stop(paths, safe_i):
        min_time = np.inf
        min_candidates = []
        for uid, path in paths.items():
            if len(path) == safe_i[uid]: continue
            if path[safe_i[uid]]['landing_time'] < min_time:
                min_time = path[safe_i[uid]]['landing_time']
                min_candidates = [(uid, safe_i[uid])]
            elif path[safe_i[uid]]['landing_time'] == min_time:
                min_candidates.append((uid, safe_i[uid]))
        max_time = -np.inf
        next_stop = None
        for uid, i in min_candidates:
            if paths[uid][i]['takeoff_time'] > max_time:
                max_time = paths[uid][i]['takeoff_time']
                next_stop = (uid, i)
        return next_stop

    @staticmethod
    def find_conflicts(paths, safe_i, uid, i):
        conflicts = []
        for u2, path in paths.items():
            if u2 == uid:
                continue
            i2 = safe_i[u2]
            while i2 < len(path) and path[i2]['landing_time'] < paths[uid][i]['takeoff_time']:
                if path[i2]['station'] == paths[uid][i]['station']:
                    conflicts.append((u2, i2))
                i2 += 1
        return conflicts

    @staticmethod
    def conflict_offset(paths, uid, i, uid2, i2):
        # Fix uid and delay uid2
        return paths[uid][i]['takeoff_time'] - paths[uid2][i2]['landing_time']

    def solve_conflict(self, paths, times, conflict_stay, conflicts):
        uid, i = conflict_stay
        # Select the conflict whose drone has the shortest completion time (it will be delayed)
        conflict = min(conflicts, key=lambda x: times[x[0]])
        uid2, i2 = conflict
        # Delay the drone
        offset = self.conflict_offset(paths, uid, i, uid2, i2)
        assert offset >= 0
        for s in paths[uid2][i2:]:
            if s['landing_time']: s['landing_time'] += offset
            if s['takeoff_time']: s['takeoff_time'] += offset
            s['available'] += offset
        times[uid2] += offset

    def remove_conflicts(self, schedule):
        conflict_count = 0
        paths, times = self.get_paths(schedule)
        total_time = max([times[u['id']] for u in self.problem.U])  # The problem's makespan, as computed from the paths
        assert abs(total_time - self.problem.makespan(schedule)) < 1e-10
        # Count of safe stops for each drone
        safe_i = {u['id']: 0 for u in self.problem.U}
        for u in self.problem.U:
            if len(paths[u['id']]) == 1:
                safe_i[u['id']] = 1
        # We select the next stop in all paths that has the earliest landing time and is the longest among the earliest
        next_stop = self.next_longest_earliest_stop(paths, safe_i)
        while next_stop is not None:
            uid, i = next_stop
            # Find conflicts with the selected stop
            conflicts = self.find_conflicts(paths, safe_i, uid, i)
            if len(conflicts) == 0:
                safe_i[uid] = i + 1
                if paths[uid][i + 1]['takeoff_time'] is None:
                    safe_i[uid] += 1  # The drone is currently at the home station
            else:
                conflict_count += 1
                conflicts = self.find_conflicts(paths, safe_i, uid, i)
                self.solve_conflict(paths, times, next_stop, conflicts)
            next_stop = self.next_longest_earliest_stop(paths, safe_i)
        total_time = max([times[u['id']] for u in self.problem.U])
        return paths, total_time, conflict_count

    def check_schedule(self, schedule):
        count_delivery = [0 for _ in range(len(self.problem.P))]
        for u, path in schedule.items():
            for p in path:
                idx_p = self.problem.P.index(p)
                count_delivery[idx_p] += 1
        for i in range(len(count_delivery)):
            if count_delivery[i] != 1:
                raise Exception(f"Not all deliveries are scheduled exactly once")
        return True