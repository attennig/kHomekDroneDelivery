"""
This script is used to plot the results of the experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FixedLocator
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Plot output')
parser.add_argument('-simulation', type=str, help='Experiment type', choices=['dynamic', 'static'])
args = parser.parse_args()

LABEL_SIZE = 28
LEGEND_SIZE = 28

max_seed = 10

experiment_type = args.simulation
folder = f"out/{experiment_type}"
if not os.path.exists(f"{folder}/plot"):
    os.makedirs(f"{folder}/plot")

# List all the files in the directory data/out
os.listdir(folder)

# Load all the files in the directory data/out as json
data = {
    "RedMat": {},
    "MILP": {},
    "MILP_opt": {},
    "MILPunbounded": {},
    "UADP": {},
    "UADPGR": {},
    "GR": {},
    "RR": {},
    "JSIPM": {}

}
delta_max = {}

colors = ["#9B5DE5",
          "#F15BB5",
          "#FEE440",
          "#FF6978",
          "#00BBF9",
          "#00F5D4",
          "#FFBE0A"]

color_blind_friendly = [
    "#882255",
    "#AA4499",
    "#CC6677",
    "#DDCC77",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#332288"]

IBM_color_blind_friendly = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000"
]
PLOT_DICT = {
    "RedMat": {
        "hatch": "",
        "markers": "s",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[4],
        "mfc": IBM_color_blind_friendly[4],
        "label": "RedMat",
    },
    "MILP": {
        "hatch": "",
        "markers": "o",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[3],
        "mfc": IBM_color_blind_friendly[3],
        "label": "kHkD",
    },
    "MILP_opt": {
        "hatch": "",
        "markers": "*",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[3],
        "mfc": IBM_color_blind_friendly[3],
        "label": "gurobi opt",
    },
    "MILPunbounded": {
        "hatch": "",
        "markers": "*",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[3],
        "mfc": IBM_color_blind_friendly[3],
        "label": "kHkD",
    },
    "JSIPM": {
        "hatch": "",
        "markers": "^",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[2],
        "mfc": IBM_color_blind_friendly[2],
        "label": "LPT",
    },
    "RR": {
        "hatch": "",
        "markers": "D",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[1],
        "mfc": IBM_color_blind_friendly[1],
        "label": "RR",
    },
    "UADP": {
        "hatch": "",
        "markers": "X",
        "linestyle": "-",
        "color": IBM_color_blind_friendly[0],
        "mfc": IBM_color_blind_friendly[0],
        "label": "Pei et al."
    },
    "UADPGR": {
        "hatch": "",
        "markers": "P",
        "linestyle": "--",
        "color": IBM_color_blind_friendly[0],
        "mfc": IBM_color_blind_friendly[0],
        "label": "UADP-GR",
    },

    "GR": {
        "hatch": "",
        "markers": "s",
        "linestyle": "-",
        "color": color_blind_friendly[4],
        "mfc": color_blind_friendly[4],
        "label": "P&D",
    },
}

for subfolder in os.listdir(folder):
    if subfolder[0] == '.': continue
    if subfolder == "plot": continue
    experiement_properties = subfolder.split('_')
    if experiement_properties[1][-1] == 'D':
        type = "D"
        stations = int(experiement_properties[1][:-1])
    else:
        type = ""
        stations = int(experiement_properties[1])

    drones = int(experiement_properties[2][1:])
    deliveries = int(experiement_properties[3][1:])
    if experiment_type == "dynamic":
        step = int(experiement_properties[4][1:])
        rate = float(experiement_properties[5][1:])
    for file in os.listdir(f"{folder}/{subfolder}"):
        with open(f'{folder}/{subfolder}/{file}') as f:
            file_name = file.split('.')[0]
            properties = file_name.split('_')
            algorithm = properties[2]
            seed = int(properties[1])
            if len(properties) > 3:
                continue
            DT = json.load(f)
            dt = DT["output"]
            dt_time = DT["times"]
            if dt: dt.update(dt_time)
            if experiment_type == "dynamic":
                key = (stations, drones, deliveries, seed, type, step, rate)
            else:
                key = (stations, drones, deliveries, seed, type)
            data[algorithm][key] = dt


@ticker.FuncFormatter
def major_formatter(x, pos):
    if x < 500:
        return x
    if x % 10 ** 3 == 0:
        tick_label = f'{int(x / 10 ** 3)}K'
    else:
        tick_label = f'{x / 10 ** 3}K'
    return tick_label


def plot(algorithms: list,
         y_data: dict,
         y_data_std: dict,
         x_ticks,
         metric: str,
         log=False,
         x_label="Number of drones",
         marker_size=10,
         expeeriment_name="experiment",
         ylim=None,
         appendix=""):
    """
    This method has the ONLY responsibility to plot data
    @param y_data_std:
    @param y_data:
    @param algorithm:
    @param type:
    @return:
    """
    print(f"Plotting {expeeriment_name}, {metric}")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6.5))
    ax1.xaxis.set_major_locator(FixedLocator(x_ticks))

    # rewrite this for n algorithms
    for i in range(len(algorithms)):
        print(f"Algorithm: {algorithms[i]}")
        print(f"y_data: {y_data[algorithms[i]]}")
        ax1.errorbar(
            x=x_ticks,
            y=y_data[algorithms[i]],
            yerr=y_data_std[algorithms[i]],
            label=PLOT_DICT[algorithms[i]]["label"] + " " + appendix,
            color=PLOT_DICT[algorithms[i]]["color"],
            marker=PLOT_DICT[algorithms[i]]["markers"],
            linestyle=PLOT_DICT[algorithms[i]]["linestyle"],
            linewidth=0.0,
            markersize=marker_size,
            capsize=5,
            elinewidth=1,
            mfc=PLOT_DICT[algorithms[i]]["mfc"])

    ax1.set_xlabel(xlabel=x_label, fontsize=LABEL_SIZE)
    ax1.set_ylabel(ylabel=metric, fontsize=LABEL_SIZE)
    if log: plt.yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
    ax1.set_ylim(ymin=ylim)
    ax1.yaxis.set_major_formatter(major_formatter)
    plt.legend(
        fancybox=True,
        framealpha=0.5,
        ncol=1,
        handletextpad=0.1,
        columnspacing=0.7,
        prop={'size': LEGEND_SIZE})
    plt.grid(linewidth=0.3)
    plt.tight_layout()
    metric_name_file = metric.replace(' ', '_')
    plt.savefig(f"{folder}/plot/{expeeriment_name}_{metric_name_file}.pdf", format='pdf')
    plt.clf()
    plt.close()


def plot_drones_on_x(metric_field="makespan",
                     metric_name="Makespan (s)",
                     ns=None,
                     nd=None,
                     drones=[],
                     type="D",
                     seeds=[],
                     algorithms=[]):
    plot_drones = {key: [] for key in algorithms}
    plot_drones_std = {key: [] for key in algorithms}

    for nu in drones:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[4] == type and key[3] in seeds]
            datapoints = [data[algorithm][key][metric_field] for key in match if data[algorithm][key] != None]
            if len(datapoints) == 0:
                plot_drones[algorithm].append(np.nan)
                plot_drones_std[algorithm].append(np.nan)
                continue
            plot_drones[algorithm].append(np.mean(datapoints))
            plot_drones_std[algorithm].append(np.std(datapoints))
    plot(algorithms=algorithms, y_data=plot_drones, y_data_std=plot_drones_std, x_ticks=drones, metric=metric_name,
         expeeriment_name=f"n_{ns}{type}_Ux_D{nd}", ylim=0)
    return


def plot_delievries_on_x(metric_field="makespan",
                         metric_name="Makespan (s)",
                         ns=None,
                         deliveries=[],
                         nu=None,
                         type="D",
                         seeds=[],
                         algorithms=[]):
    plot_deliveries = {key: [] for key in algorithms}
    plot_deliveries_std = {key: [] for key in algorithms}
    for nd in deliveries:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[4] == type and key[3] in seeds]
            datapoints = [data[algorithm][key][metric_field] for key in match]
            if len(datapoints) == 0:
                plot_deliveries[algorithm].append(np.nan)
                plot_deliveries_std[algorithm].append(np.nan)
                continue
            plot_deliveries[algorithm].append(np.mean(datapoints))
            plot_deliveries_std[algorithm].append(np.std(datapoints))
    plot(algorithms, plot_deliveries, plot_deliveries_std, deliveries, metric_name, x_label="Number of Deliveries",
         expeeriment_name=f"n_{ns}{type}_U{nu}_Dx", ylim=0)
    return


def plot_delievries_on_x_improvement(algorithms=[], ns=None, deliveries=[], nu=None, type="D", seeds=[]):
    metric_field = "makespan"
    metric_name = f"Makespan improvement (%)"
    plot_deliveries = {key: [] for key in data.keys()}
    plot_deliveries_std = {key: [] for key in data.keys()}
    legend = [algo for algo in algorithms if algo != "RedMat"]

    for nd in deliveries:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[4] == type and key[3] in seeds]
            datapoints = [
                100 * (data[algorithm][key][metric_field] - data["RedMat"][key][metric_field]) / data[algorithm][key][
                    metric_field] for key in match if data[algorithm][key] != None]
            if len(datapoints) == 0:
                plot_deliveries[algorithm].append(np.nan)
                plot_deliveries_std[algorithm].append(np.nan)
                continue
            plot_deliveries[algorithm].append(np.mean(datapoints))
            plot_deliveries_std[algorithm].append(np.std(datapoints))
    plot(algorithms=legend, y_data=plot_deliveries, y_data_std=plot_deliveries_std, x_ticks=deliveries,
         metric=metric_name, x_label="Number of Deliveries", expeeriment_name=f"n_{ns}{type}_U{nu}_Dx")
    return


def plot_drones_on_x_improvement(algorithms=[], ns=None, nd=None, drones=[], type="D", seeds=[]):
    metric_field = "makespan"
    metric_name = f"Makespan improvement (%)"
    print(algorithms)
    plot_drones = {key: [] for key in data.keys()}
    plot_drones_std = {key: [] for key in data.keys()}
    legend = [algo for algo in algorithms if algo != "RedMat"]
    for algorithm in legend:
        for nu in drones:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[4] == type and key[3] in seeds]
            datapoints = [
                100 * (data[algorithm][key][metric_field] - data["RedMat"][key][metric_field]) / data[algorithm][key][
                    metric_field] for key in match if data[algorithm][key] != None]
            if len(datapoints) == 0:
                plot_drones[algorithm].append(np.nan)
                plot_drones_std[algorithm].append(np.nan)
                continue
            plot_drones[algorithm].append(np.mean(datapoints))
            plot_drones_std[algorithm].append(np.std(datapoints))
    plot(algorithms=legend, y_data=plot_drones, y_data_std=plot_drones_std, x_ticks=drones, metric=metric_name,
         expeeriment_name=f"n_{ns}{type}_Ux_D{nd}")
    return


def plot_dynamic_rates_on_x(metric_field="makespan",
                            metric_name="Makespan (s)",
                            aggregator=None,
                            ns=50,
                            nd=500,
                            nu=20,
                            type="D",
                            seeds=[],
                            algorithms=[],
                            rates=[],
                            step=30):
    plot_ = {key: [] for key in data.keys()}
    plot_std_ = {key: [] for key in data.keys()}
    for rate in rates:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[3] in seeds and key[4] == type and key[
                         5] == step and key[6] == rate]
            if aggregator:
                datapoints = [data[algorithm][key][metric_field][aggregator] for key in match]
            else:
                datapoints = [data[algorithm][key][metric_field] for key in match]
            if len(datapoints) == 0:
                plot_[algorithm].append(np.nan)
                plot_std_[algorithm].append(np.nan)
                continue
            plot_[algorithm].append(np.mean(datapoints))
            plot_std_[algorithm].append(np.std(datapoints))

    plot(algorithms, plot_, plot_std_, rates, metric_name, x_label="Arrival rate (dpm)",
         expeeriment_name=f"n_{ns}{type}_U{nu}_D{nd}_S{step}_Rx", ylim=0, appendix="Dyn")
    return


def plot_dynamic_steps_on_x(metric_field="makespan",
                            metric_name="Makespan (s)",
                            aggregator=None,
                            ns=50,
                            nd=500,
                            nu=20,
                            type="D",
                            seeds=[],
                            algorithms=[],
                            rate=1,
                            steps=[]):
    plot_ = {key: [] for key in data.keys()}
    plot_std_ = {key: [] for key in data.keys()}
    for step in steps:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[3] in seeds and key[4] == type and key[
                         5] == step and key[6] == rate]
            if aggregator:
                datapoints = [data[algorithm][key][metric_field][aggregator] for key in match]
            else:
                datapoints = [data[algorithm][key][metric_field] for key in match]
            if len(datapoints) == 0:
                plot_[algorithm].append(np.nan)
                plot_std_[algorithm].append(np.nan)
                continue
            plot_[algorithm].append(np.mean(datapoints))
            plot_std_[algorithm].append(np.std(datapoints))
    plot(algorithms, plot_, plot_std_, steps, metric_name, x_label="Pooling time (min)",
         expeeriment_name=f"n_{ns}{type}_U{nu}_D{nd}_Sx_R{rate}", ylim=0, appendix="Dyn")
    return


def plot_dynamic_deliveries_on_x(metric_field="makespan",
                                 metric_name="Makespan (s)",
                                 aggregator=None,
                                 ns=50,
                                 deliveries=[],
                                 nu=20,
                                 type="D",
                                 seeds=[],
                                 algorithms=[],
                                 rate=0.5,
                                 step=30):
    plot_ = {key: [] for key in data.keys()}
    plot_std_ = {key: [] for key in data.keys()}
    for nd in deliveries:
        for algorithm in algorithms:
            match = [key for key in data[algorithm].keys() if
                     key[0] == ns and key[1] == nu and key[2] == nd and key[3] in seeds and key[4] == type and key[
                         5] == step and key[6] == rate]
            if aggregator:
                datapoints = [data[algorithm][key][metric_field][aggregator] for key in match]
            else:
                datapoints = [data[algorithm][key][metric_field] for key in match]

            if len(datapoints) == 0:
                plot_[algorithm].append(np.nan)
                plot_std_[algorithm].append(np.nan)
                continue
            plot_[algorithm].append(np.mean(datapoints))
            plot_std_[algorithm].append(np.std(datapoints))
    plot(algorithms, plot_, plot_std_, deliveries, metric_name, x_label="Number of deliveries",
         expeeriment_name=f"n_{ns}{type}_U{nu}_Dx_S{step}_R{rate}", ylim=0, appendix="Dyn")
    return


def plot_static_experiments():
    EXPERIMENTS_var_drones = {
        "small-D": {
            "seeds": list(range(1, 6)),
            "stations": [10],
            "drones": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "deliveries": [30],
            "algorithms": ["RedMat", "MILPunbounded", "JSIPM", "RR", "UADP"]  # "MILP", "MILP_opt"
        },
        "small-S": {
            "seeds": list(range(1, 6)),
            "stations": [10],
            "drones": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "deliveries": [30],
            "algorithms": ["RedMat", "MILP", "JSIPM", "RR"]  #MILPunbounded
        },
        "large-D": {
            "seeds": list(range(1, 11)),
            "stations": [50],
            "drones": list(range(5, 45, 5)),
            "deliveries": [100],
            "algorithms": ["RedMat", "JSIPM", "RR", "UADP"]
        },
        "large-S": {
            "seeds": list(range(1, 11)),
            "stations": [50],
            "drones": list(range(5, 45, 5)),
            "deliveries": [100],
            "algorithms": ["RedMat", "JSIPM", "RR"]
        }
    }
    EXPERIMENTS_var_deliveries = {
        "small-D": {
            "seeds": list(range(1, 6)),
            "stations": [10],
            "drones": [4],
            "deliveries": [30, 40, 50, 60, 70, 80, 90, 100],  # 10, 20,
            "algorithms": ["RedMat", "MILPunbounded", "JSIPM", "RR", "UADP"]  # "MILP", "MILP_opt"
        },
        "small-S": {
            "seeds": list(range(1, 6)),
            "stations": [10],
            "drones": [4],
            "deliveries": [30, 40, 50, 60, 70, 80, 90, 100],  # 10, 20,
            "algorithms": ["RedMat", "MILP", "JSIPM", "RR"]
        },
        "large-D": {
            "seeds": list(range(1, 11)),
            "stations": [50],
            "drones": [20],
            "deliveries": list(range(50, 201, 25)),
            "algorithms": ["RedMat", "JSIPM", "RR", "UADP"]
        },
        "large-S": {
            "seeds": list(range(1, 11)),
            "stations": [50],
            "drones": [20],
            "deliveries": list(range(50, 201, 25)),
            "algorithms": ["RedMat", "JSIPM", "RR"]
        }
    }
    scenarios = []
    for experiment, parameters in EXPERIMENTS_var_drones.items():
        size = experiment.split("-")[0]
        type = "" if experiment.split("-")[-1] == "S" else "D"
        for ns in parameters["stations"]:
            for nd in parameters["deliveries"]:
                for nu in parameters["drones"]:
                    if type == "D" and size == "small": scenarios.append((ns, nd, nu))
                plot_drones_on_x_improvement(algorithms=parameters["algorithms"], ns=ns, nd=nd,
                                             drones=parameters["drones"], type=type, seeds=parameters["seeds"])
                for field, name in [("makespan", "Makespan")]:  # , ("runtime", "Runtime (s)")]:
                    plot_drones_on_x(
                        metric_field=field,
                        metric_name=name,
                        ns=ns,
                        nd=nd,
                        drones=parameters["drones"],
                        type=type,
                        seeds=parameters["seeds"],
                        algorithms=parameters["algorithms"]
                    )

    for experiment, parameters in EXPERIMENTS_var_deliveries.items():
        size = experiment.split("-")[0]
        type = "" if experiment.split("-")[-1] == "S" else "D"
        for ns in parameters["stations"]:
            for nu in parameters["drones"]:
                for nd in parameters["deliveries"]:
                    if type == "D" and size == "small": scenarios.append((ns, nd, nu))
                plot_delievries_on_x_improvement(algorithms=parameters["algorithms"], ns=ns, nu=nu,
                                                 deliveries=parameters["deliveries"], type=type,
                                                 seeds=parameters["seeds"])
                for field, name in [("makespan", "Makespan")]:  # , ("runtime", "Runtime (s)")]:
                    plot_delievries_on_x(
                        metric_field=field,
                        metric_name=name,
                        ns=ns,
                        deliveries=parameters["deliveries"],
                        nu=nu,
                        type=type,
                        seeds=parameters["seeds"],
                        algorithms=parameters["algorithms"]
                    )


def plot_dynamic_experiments():
    EXPERIMENTS = {
        "large-D": {
            "seeds": list(range(1, 6)),
            "stations": [50],
            "type": "D",
            "drones": [20],
            "deliveries": [100, 200, 300, 400, 500],
            "algorithms": ["RedMat", "UADP"],  # "JSIPM",
        }
    }
    rates, step = [0.25, 0.5, 0.75, 1.0, 1.25], 30
    rate, steps = 0.5, [10, 15, 20, 25, 30]
    metrics = [("makespan", "Makespan (s)", None), ("delivery_times", "Mean delivery time (s)", "average"),
               ("busy_times", "Total drone busy time (s)", "total")]
    for metric_field, metric_name, aggregator in metrics:
        for experiment, parameters in EXPERIMENTS.items():
            for ns in parameters["stations"]:
                for nu in parameters["drones"]:
                    plot_dynamic_deliveries_on_x(metric_field=metric_field, metric_name=metric_name,
                                                 aggregator=aggregator,
                                                 ns=ns, deliveries=parameters["deliveries"], nu=nu,
                                                 type=parameters["type"],
                                                 seeds=parameters["seeds"], algorithms=parameters["algorithms"],
                                                 rate=rate, step=step)
                    plot_dynamic_rates_on_x(metric_field=metric_field, metric_name=metric_name, aggregator=aggregator,
                                            ns=ns, nd=parameters["deliveries"][-1], nu=nu, type=parameters["type"],
                                            seeds=parameters["seeds"], algorithms=parameters["algorithms"],
                                            rates=rates, step=step)
                    plot_dynamic_steps_on_x(metric_field=metric_field, metric_name=metric_name, aggregator=aggregator,
                                            ns=ns, nd=parameters["deliveries"][-1], nu=nu, type=parameters["type"],
                                            seeds=parameters["seeds"], algorithms=parameters["algorithms"],
                                            rate=rate, steps=steps)


if __name__ == '__main__':
    if experiment_type == "static":
        plot_static_experiments()
    elif experiment_type == "dynamic":
        plot_dynamic_experiments()
    else:
        print("Invalid experiment type")
        exit(1)
