from src.Problem import ProblemInstance
from src.solvers.ReductionMatching import ReductionMatching
from src.solvers.UADP import UADP
from src.solvers.MILP import MILP
from src.solvers.Baseline import Baseline
import json
import numpy as np
import os
import time as t_lib
import argparse

parser = argparse.ArgumentParser(description='Solve Drone Delivery problem.')
parser.add_argument('-ns', type=int, help='Number of stations')
parser.add_argument('-nu', type=int, help='Number of drones')
parser.add_argument('-nd', type=int, help='Number of deliveries')
parser.add_argument('-instance', type=int, help='experiment instance', default=0)
parser.add_argument('-parameters', type=str, help='path to parameters file',
                    default='data/parameters/default.json')
parser.add_argument('-algorithm', type=str,
                    help='Selected algorithm in ["RedMat", "MILP", "MILPunboundend", "UADP", "UADPGR", "RR", "JSIPM"]')
parser.add_argument('-simulation', type=str, help='Dynamic or Static', choices=['dynamic', 'static', 'dimensioning'],
                    default='static')
parser.add_argument('-step', type=int, help='Step to advance the time at each iteration', default=15)
parser.add_argument('-drone_model', type=str, help='Drone model', default='data/drone_models/octocopter.json')
parser.add_argument('-type', type=str, help='Type of instance', default='')
parser.add_argument('-dpm', type=float, help='Delivery per minute', default=1)

args = parser.parse_args()

NEW_DELIVERIES_RATE = args.dpm  # deliveries per minute
WIND_CHANGE_RATE = 60  # minutes


def load_entities(filename, n):
    with open(filename, 'r') as f:
        entities = json.load(f)[:n]
    return entities

def load_solver(algorithm, problem):
    if algorithm == "RedMat":
        solver = ReductionMatching(problem, args=args)
    elif algorithm == "MILP" or algorithm == "MILPunbounded":
        solver = MILP(problem, args=args)
    elif algorithm == "UADP":
        solver = UADP(problem, args=args)
    elif algorithm == "UADPGR":
        solver = UADP(problem, only_greedy=True, args=args)
    elif args.algorithm in ["RR", "JSIPM"]:
        solver = Baseline(problem, args.algorithm, seed=args.instance)
    else:
        raise Exception(f"Sorry, no algorithm {algorithm} available")
    return solver


def dynamic_simulation(algorithm, stations, drones, deliveries, wind, param, step):
    assert algorithm in ["RedMat", "MILP", "UADP", "UADPGR", "RR", "JSIPM"]
    step *= 60  # Convert step to seconds

    delivery_by_id = {p['id']: p for p in deliveries}
    step_outputs = []  # We store the output of each step for debugging purposes
    schedule = {u['id']: [] for u in drones}
    paths = {u['id']: [{'station': u['home'],
                        'landing_time': 0,
                        'takeoff_time': None,
                        'available': 0,
                        'delivery': None}] for u in drones}
    last_stop = {u['id']: 0 for u in drones}
    time = step
    wind_index = 0
    last_wind_change = 0
    for u in drones:
        u['start'] = u['home']
        u['availability_time'] = 0
        u['time_to_complete_running_task'] = 0
    for p in deliveries:
        p['input_time'] = None
        p['departure_time'] = None
        p['arrival_time'] = None
    non_arrived_deliveries = [p for p in deliveries]
    pending_deliveries = set()
    delivered_deliveries = set()
    rounds = 0
    sum_of_execution_times = 0
    new_schedule_time = 0
    history = []

    np.random.seed(args.instance)

    while non_arrived_deliveries or pending_deliveries:
        data = {
            "time": time,
            "pending_deliveries": list(pending_deliveries),
            "delivered_deliveries": [],
        }
        new_deliveries = []
        if non_arrived_deliveries:
            # Fetching new deliveries to be scheduled
            for m in range(step - 60, -1, -60):
                n_next = np.random.poisson(NEW_DELIVERIES_RATE)
                n_deliveries = non_arrived_deliveries[:min(n_next, len(non_arrived_deliveries))]
                for p in n_deliveries:
                    p['input_time'] = time - m
                    pending_deliveries.add(p['id'])
                new_deliveries += n_deliveries
                non_arrived_deliveries = non_arrived_deliveries[min(n_next, len(non_arrived_deliveries)):]
        data["new_deliveries"] = [p['id'] for p in new_deliveries]
        # Manage wind
        wind_change = False
        if time - last_wind_change >= WIND_CHANGE_RATE * 60:
            wind_index = (wind_index + 1) % len(wind)
            last_wind_change = time
            wind_change = True
        data["wind_change"] = wind_change
        wind_speed = float(wind[wind_index]['speed'])
        wind_direction = float(wind[wind_index]['direction'])
        data["solver"] = {}
        if new_deliveries or (wind_change and pending_deliveries):
            # Instantiate and solve static problem
            start_timestamp = t_lib.time()
            problem = ProblemInstance(stations,
                                      drones,
                                      [delivery_by_id[p] for p in pending_deliveries],
                                      wind_speed=wind_speed,
                                      wind_direction=wind_direction,
                                      param_file=param,
                                      drone_model=args.drone_model)
            solver = load_solver(algorithm, problem)
            output = solver.solve()
            last_execution_time = t_lib.time() - start_timestamp
            sum_of_execution_times += last_execution_time
            rounds += 1
            step_outputs.append(output)
            if output is None:
                print("No solution found.")
                break
            new_paths = output['paths']
            data["solver"]["runtime"] = last_execution_time
            # Deal with execution time forecasting error
            time_after_execution = time + last_execution_time
            if time_after_execution > new_schedule_time:
                delay = time_after_execution - new_schedule_time
            else:
                delay = 0
            busy_time = min([u['time_to_complete_running_task'] for u in drones])
            if delay > busy_time:
                for u in new_paths:
                    for stop in new_paths[u]:
                        stop['landing_time'] += delay
                        if stop['takeoff_time'] is not None: stop['takeoff_time'] += delay
                        stop['available'] += delay
            data["solver"]["delay"] = delay
            data["solver"]["first_available_delay"] = busy_time

            # Update paths
            for u in drones:
                if last_stop[u['id']] < len(paths[u['id']]):
                    if not algorithm == "UADP":
                        new_paths[u['id']][0]['landing_time'] = paths[u['id']][last_stop[u['id']]]['landing_time']
                    new_paths[u['id']][0]['available'] = paths[u['id']][last_stop[u['id']]]['available']
                paths[u['id']] = paths[u['id']][:last_stop[u['id']]] + new_paths[u['id']]

        # Forecast execution time
        time += step
        new_schedule_time = time + (sum_of_execution_times / rounds) if rounds else time

        # Update schedule and prepare for next iteration
        for u in drones:
            u['availability_time'] = new_schedule_time
            u['time_to_complete_running_task'] = 0
            u['start'] = paths[u['id']][min(last_stop[u['id']] + 1, len(paths[u['id']]) - 1)]['station']
            for i in range(last_stop[u['id']], len(paths[u['id']])):
                if paths[u['id']][i]['available'] > new_schedule_time:
                    u['availability_time'] = paths[u['id']][i]['available']
                    u['start'] = paths[u['id']][i]['station']
                    u['time_to_complete_running_task'] = paths[u['id']][i]['available'] - new_schedule_time
                    last_stop[u['id']] = i
                    break
                else:
                    if paths[u['id']][i]['delivery'] is not None:
                        delivery = delivery_by_id[paths[u['id']][i]['delivery']]
                        if algorithm == "UADP":
                            delivery['departure_time'] = paths[u['id']][i]['takeoff_time'] + problem.delta_l
                        else:
                            delivery['departure_time'] = paths[u['id']][i]['takeoff_time']
                        delivery['arrival_time'] = paths[u['id']][i + 1]['available'] - problem.delta_r
                        schedule[u['id']].append(delivery)
                        if delivery['id'] not in pending_deliveries:
                            print(f"Delivery {delivery['id']} was not pending.")
                        delivered_deliveries.add(delivery['id'])
                        data["delivered_deliveries"].append((delivery['id'], u['id']))
                        pending_deliveries.remove(delivery['id'])
                        last_stop[u['id']] = i + 1
                    else:
                        last_stop[u['id']] = i
                    u['start'] = paths[u['id']][i]['station']
        history.append(data)
    return delivery_by_id, {
        'schedule': schedule,
        'paths': paths,
        'makespan': max([paths[u['id']][-1]['available'] for u in drones]),
        'outputs': step_outputs
    }, {
        "history": history,
        "sum_of_execution_times": sum_of_execution_times,
        "rounds": rounds
    }


def static_simulation(algorithm, stations, drones, deliveries, current_wind, param_file):
    assert algorithm in ["RedMat", "MILP", "MILPunbounded", "UADP", "UADPGR", "RR", "JSIPM"]

    for u in drones:
        u['start'] = u['home']
        u['availability_time'] = 0
        u['time_to_complete_running_task'] = 0
    start_time = t_lib.time()
    problem = ProblemInstance(stations, drones, deliveries, wind_speed=current_wind['speed'],
                              wind_direction=current_wind['direction'],
                              param_file=param_file, drone_model=args.drone_model)
    problem_loading_time = t_lib.time()
    solver = load_solver(algorithm, problem)
    output = solver.solve()
    final_time = t_lib.time()

    print(f"Problem loading time: {problem_loading_time - start_time}")
    print(f"Solver execution time: {final_time - problem_loading_time}")
    print(f"Total time: {final_time - start_time}")

    if output:
        solver.check_schedule(output['schedule'])
        print(f"makespan: {output['makespan']} ?= {problem.makespan(output['schedule'])}")

    return problem, output, {
        "problem_loading_time": problem_loading_time - start_time,
        "solver_execution_time": final_time - problem_loading_time,
        "total_time": final_time - start_time
    }


def print_history(deliveries, history):
    for i, step in enumerate(history):
        print(f"------ Step: {i} ------ Time: {step['time']} ------")
        print(f"Arrived deliveries:")
        for p in step['new_deliveries']:
            print(f" - Delivery #{p}  [{deliveries[p]['src']} --> {deliveries[p]['dst']}]")
        print(f"Pending deliveries:")
        for p in step['pending_deliveries']:
            print(f" - Delivery #{p}  [{deliveries[p]['src']} --> {deliveries[p]['dst']}]")
        print(f"Will be delivered:")
        for p in step['delivered_deliveries']:
            print(f" - Delivery #{p[0]}  [{deliveries[p[0]]['src']} --> {deliveries[p[0]]['dst']}] (Drone #{p[1]})")
        print("The wind changed" if step['wind_change'] else "The wind did NOT change")
        if step['solver']:
            print(f"Solver runtime: {step['solver']['runtime']}")
            print(f"Solver delay: {step['solver']['delay']}")
            print(f"Solver first available delay: {step['solver']['first_available_delay']}")
        else:
            print("No solver execution")
        print('\n\n')


def analyze_dynamic_solution(paths, deliveries, times):
    print("------ Dynamic solution analysis ------\n")
    print("*** Busy time analysis ***")
    busy_times = {}
    makespan = max(paths.keys(), key=lambda k: paths[k][-1]['available'])
    print(f"Makespan: {paths[makespan][-1]['available']} (Drone #{makespan})")
    total_busy_time = 0
    for u in paths:
        print(f"--- [Drone #{u}]")
        print(f"    End time: {paths[u][-1]['available']}")
        busy_time = sum([paths[u][i + 1]['available'] - paths[u][i]['takeoff_time'] for i in range(len(paths[u]) - 1)])
        print(f"    Busy time: {busy_time}")
        total_busy_time += busy_time
        busy_times[u] = busy_time
        print()
    print(f"Total busy time: {total_busy_time}\n")
    busy_times['total'] = total_busy_time

    print("*** Delivery time analysis ***")
    total_delivery_time = 0
    delivery_times = {}
    for p in deliveries:
        delivery_time = deliveries[p]['arrival_time'] - deliveries[p]['input_time']
        print(f"--- [Delivery #{p}]")
        print(f"    Departure time: {deliveries[p]['departure_time']}")
        print(f"    Arrival time: {deliveries[p]['arrival_time']}")
        print(f"    Delivery time: {delivery_time}")
        total_delivery_time += delivery_time
        delivery_times[p] = delivery_time
        print()
    average_delivery_time = total_delivery_time / len(deliveries)
    print(f"Average delivery time: {average_delivery_time}\n")
    delivery_times['average'] = average_delivery_time

    times['busy_times'] = busy_times
    times['delivery_times'] = delivery_times


def dynamic_dimensioning(stations, drones, deliveries, wind, param_file):
    for u in drones:
        u['start'] = u['home']

    delivery_times = []
    time_to_pickup = []
    for current_wind in wind:
        problem = ProblemInstance(stations, drones, deliveries, wind_speed=current_wind['speed'],
                                  wind_direction=current_wind['direction'],
                                  param_file=param_file, drone_model=args.drone_model)
        time_to_pickup.append(np.mean(list(problem.DELTA.values())))
        for p in deliveries:
            delivery_times.append(problem.DELTA_weighted[p['id']])
    mean_delivery_time = np.mean(delivery_times) / 60
    max_pickup_time = np.max(time_to_pickup) / 60
    TME = mean_delivery_time + max_pickup_time
    mu = 1 / TME  # delivery per minute

    return {
        "mean_delivery_time": mean_delivery_time,
        "max_pickup_time": max_pickup_time,
        "TME": TME,
        "mu": mu
    }


if __name__ == '__main__':
    print(f"Running {args.algorithm} on instance {args.instance}. Mode: {args.simulation}.")
    print(f"{args.ns}{args.type} stations, {args.nu} drones, {args.nd} deliveries.")
    if args.simulation == 'dynamic':
        print(f"Step: {args.step} seconds, DPM: {args.dpm}.")
    print(f"Parameters file: {args.parameters}, Step: {args.step}.")

    if not os.path.exists("out"):
        os.makedirs("out")
    if not os.path.exists(f"out/{args.simulation}"):
        os.makedirs(f"out/{args.simulation}")
    if args.simulation == 'dynamic':
        output_path = f"out/{args.simulation}/n_{args.ns}{args.type}_U{args.nu}_D{args.nd}_S{args.step}_R{args.dpm}"
    else:
        output_path = f"out/{args.simulation}/n_{args.ns}{args.type}_U{args.nu}_D{args.nd}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    stations = load_entities(f"data/experiments/n_{args.ns}{args.type}/e_{args.instance}/facilities.json", args.ns)
    drones = load_entities(f"data/experiments/n_{args.ns}{args.type}/e_{args.instance}/fleet.json", args.nu)
    deliveries = load_entities(f"data/experiments/n_{args.ns}{args.type}/e_{args.instance}/demand.json", args.nd)
    wind = load_entities(f"data/experiments/n_{args.ns}{args.type}/e_{args.instance}/winds.json", 20)

    if args.simulation == 'dynamic':
        deliveries, output, times = dynamic_simulation(args.algorithm, stations, drones, deliveries, wind,
                                                       args.parameters, args.step)
        print_history(deliveries, times['history'])
        analyze_dynamic_solution(output['paths'], deliveries, times)
        times['deliveries'] = deliveries
    elif args.simulation == 'static':
        _, output, times = static_simulation(args.algorithm, stations, drones, deliveries, wind[0], args.parameters)
    else:
        output = dynamic_dimensioning(stations, drones, deliveries, wind, args.parameters)
        with open(f"{output_path}/e_{args.instance}_dimensioning.json", 'w') as f:
            json.dump(output, f)
        exit()

    if output is not None:
        print(f"Algorithm's makespan: {output['makespan']}")

    solution = {'algorithm': args.algorithm, 'output': output, "times": times}

    print("saving output")
    with open(
            f"{output_path}/e_{args.instance}_{args.algorithm}.json",
            'w') as f:
        json.dump(solution, f)
