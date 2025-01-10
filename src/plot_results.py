import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
# import copy


def timesteps_to_datetime(timesteps_np: np.ndarray) -> np.datetime64:
    time_delta = np.timedelta64(900, "s") * timesteps_np
    # calculate the datetime corresponding to `timestep`
    datetimes = np.datetime64("2020-01-01 00:00:00") + time_delta
    return datetimes


def plot_sim_run(rundir, days: list = None):
    # assumed contents of the directory:
    # agents.json
    # bus_vm_pu.json
    # line_load.json

    plot_line_load(os.path.join(rundir, "line_load.json"), rundir, days)
    plot_vm_pu(os.path.join(rundir, "bus_vm_pu.json"), rundir, days)
    plot_agents(os.path.join(rundir, "agents.json"), rundir, days)


def plot_line_load(line_load_file, rundir, days):
    line_res = get_line_load(line_load_file=line_load_file)

    # plot of line loading for the selected day(s)
    if days is not None:
        fig = plt.figure(figsize=(10, 4))
        plt.xlabel("time")
        plt.ylabel("Loading, in %")
        plt.xlim(days[0] * 96, (days[-1] + 1) * 96)
        for name in line_res["df"].columns:
            plt.plot(line_res["df"][name], label=name)
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "days_line_loading.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

    # boxplot for all lines
    fig = plt.figure(figsize=(25, 3))
    plt.title("Box plot of lines loading")
    plt.ylabel("Loading, in %")
    line_res["df"].boxplot()
    plt.xticks(rotation=45, ha="right")
    fig.get_axes()[0].set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "line_boxpl.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # number of violations plus overload work
    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(line_res["df"].columns, line_res["num_ovl_list"], width=0.8)
    for i, value in enumerate(line_res["num_ovl_list"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("number of violated steps")
    plt.ylim((0, max(1.1 * max(line_res["num_ovl_list"]), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "line_num_viol.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # printing of statistics
    print(
        f"LINE: Max {line_res['max_val']} , Min {line_res['min_val']} "
        f", #Ovl {line_res['num_ovl_sum']}, Ovl work {line_res['ovl_work_kWh']} kWh"
    )


def get_line_load(line_load_file) -> dict:
    res_dict = {}

    with open(line_load_file, "r") as f:
        data = json.load(f)
        data_res = {}

        # go by all lines and sort the results in lists
        for line, results_list in data.items():
            data_res[line] = []
            for idx_day, res_list_day in enumerate(results_list):
                for val in res_list_day:
                    data_res[line].append(val)
        res_dict["df"] = pd.DataFrame(data_res)

        # number of violations plus overload work
        threshold = 100
        ovl_work_kWh = 0
        n_overloads_list = []
        for name in data_res.keys():
            n_overloads_list.append(sum(res_dict["df"][name] > threshold))
            ovl_work_kWh += round(
                sum(
                    (res_dict["df"][name][res_dict["df"][name] > threshold] - threshold)
                    * 400
                    / 4
                )
            )
        res_dict["num_ovl_list"] = n_overloads_list
        res_dict["ovl_work_kWh"] = ovl_work_kWh

        res_dict["max_val"] = round(res_dict["df"].max().max(), 3)
        res_dict["min_val"] = round(res_dict["df"].min().min(), 3)
        res_dict["mean_val"] = round(res_dict["df"].mean().mean(), 3)
        res_dict["num_ovl_sum"] = round(sum(n_overloads_list), 3)

    return res_dict


def plot_vm_pu(vm_pu_file, rundir, days):
    bus_res = get_vm_pu(vm_pu_file=vm_pu_file)

    # plot of line loading for the selected day(s)
    if days is not None:
        fig = plt.figure(figsize=(10, 4))
        plt.xlabel("time")
        plt.ylabel("Voltage magnitude, in p.u.")
        plt.xlim(days[0] * 96, (days[-1] + 1) * 96)
        for name in bus_res["df"].columns:
            plt.plot(bus_res["df"][name], label=name)
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "days_bus_vm_pu.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

    # boxplot for all buses
    fig = plt.figure(figsize=(25, 3))
    plt.title("Box plot of bus voltage magnitude")
    plt.ylabel("Voltage magnitude, in p.u.")
    bus_res["df"].boxplot()
    plt.xticks(rotation=45, ha="right")
    fig.get_axes()[0].set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "bus_boxpl.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # number of violations
    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(bus_res["df"].columns, bus_res["num_ovl_up_list"], width=0.8)
    plt.bar(bus_res["df"].columns, bus_res["num_ovl_lw_list"], width=0.8)
    for i, value in enumerate(bus_res["num_ovl_up_list"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(bus_res["num_ovl_lw_list"]):
        plt.text(i, value - 1, str(value), ha="center", va="bottom")
    plt.ylabel("number of violated steps")
    plt.ylim(
        (
            min(1.1 * min(bus_res["num_ovl_lw_list"]), -1),
            max(1.1 * max(bus_res["num_ovl_up_list"]), 1),
        )
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "bus_num_viol.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # printing of statistics
    print(
        f"BUS: Max {bus_res['max_val']} , Min {bus_res['min_val']} "
        f", #O_up {sum(bus_res['num_ovl_up_list'])} , #O_lw {sum(bus_res['num_ovl_lw_list'])}, "
        f"Sum vm over {bus_res['ovl_sum_ov']}, sum vm under {bus_res['ovl_sum_lw']}"
    )


def get_vm_pu(vm_pu_file) -> dict:
    res_dict = {}

    with open(vm_pu_file, "r") as f:
        data = json.load(f)
        data_res = {}

        # go by all buses and sort the results in lists
        for bus, results_list in data.items():
            data_res[bus] = []
            for idx_day, res_list_day in enumerate(results_list):
                for val in res_list_day:
                    data_res[bus].append(val)
        res_dict["df"] = pd.DataFrame(data_res)

        # number of violations
        threshold_up = 1.05
        threshold_lw = 0.95
        sum_vm_ov = 0
        sum_vm_un = 0
        n_up_list = []
        n_lw_list = []
        for name in data_res.keys():
            n_up_list.append(sum(res_dict["df"][name] > threshold_up))
            n_lw_list.append(-sum(res_dict["df"][name] < threshold_lw))
            sum_vm_ov += round(
                sum(
                    res_dict["df"][name][res_dict["df"][name] > threshold_up]
                    - threshold_up
                ),
                4,
            )
            sum_vm_un += round(
                sum(
                    threshold_lw
                    - res_dict["df"][name][res_dict["df"][name] < threshold_lw]
                ),
                4,
            )

        res_dict["num_ovl_up_list"] = n_up_list
        res_dict["num_ovl_lw_list"] = n_lw_list
        res_dict["ovl_sum_ov"] = round(sum_vm_ov, 4)
        res_dict["ovl_sum_lw"] = round(sum_vm_un, 4)

        res_dict["max_val"] = round(res_dict["df"].max().max(), 4)
        res_dict["min_val"] = round(res_dict["df"].min().min(), 4)
        res_dict["mean_val"] = round(res_dict["df"].mean().mean(), 4)

    return res_dict


def plot_agents(agents_file, rundir, days):
    agents_res = get_agents_res(agents_file=agents_file)

    # plot of residual power for the selected day(s)
    if days is not None:
        fig = plt.figure(figsize=(10, 4))
        plt.xlabel("time")
        plt.ylabel("Power, in kW")
        plt.xlim(0, len(days) * 96)
        for name, profile in agents_res["df"].items():
            plt.plot(profile, label=name)
        # plt.legend()
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "days_agents_p.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

    # boxplot for all agents
    fig = plt.figure(figsize=(25, 3))
    plt.title("Box plot of agents power")
    plt.ylabel("Power, in kW")
    agents_res["df"].boxplot()
    plt.xticks(rotation=45, ha="right")
    fig.get_axes()[0].set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "agents_boxpl.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # Energy sums of agents
    fig, ax = plt.subplots(layout="constrained", figsize=(15, 4))
    plt.bar(agents_res["df"].keys(), agents_res["amount_e_list"], width=0.8)
    for i, value in enumerate(agents_res["amount_e_list"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("Energy, in kWh")
    # plt.ylim((0, max(1.1 * max(amount_e), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "agents_energy.png"),
        dpi=100,
        format="png",
        bbox_inches="tight",
    )
    fig, ax = plt.subplots(layout="constrained", figsize=(15, 4))
    plt.bar(agents_res["df"].keys(), agents_res["amount_edemand_list"], width=0.8)
    plt.bar(agents_res["df"].keys(), agents_res["amount_efeedin_list"], width=0.8)
    for i, value in enumerate(agents_res["amount_edemand_list"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(agents_res["amount_efeedin_list"]):
        plt.text(i, value - 1, str(value), ha="center", va="bottom")
    plt.ylabel("Energy demanded from/fed into grid, in kWh")
    plt.ylim(
        (
            min(1.1 * min(agents_res["amount_efeedin_list"]), -1),
            max(1.1 * max(agents_res["amount_edemand_list"]), 1),
        )
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "agents_energy_sep.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    fig, ax = plt.subplots(layout="constrained", figsize=(15, 4))
    plt.bar(agents_res["df"].keys(), agents_res["amount_e_cons"], width=0.8)
    plt.bar(agents_res["df"].keys(), agents_res["amount_e_gen"], width=0.8)
    for i, value in enumerate(agents_res["amount_e_cons"]):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(agents_res["amount_e_gen"]):
        plt.text(i, value - 1, str(value), ha="center", va="bottom")
    plt.ylabel("Energy consumed/generated, in kWh")
    plt.ylim(
        (
            min(1.1 * min(agents_res["amount_e_gen"]), -1),
            max(1.1 * max(agents_res["amount_e_cons"]), 1),
        )
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join(rundir, "agents_energy_sep_test.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # printing of statistics
    print(
        f"AGENTS: MeanE {agents_res['mean_energy']} kWh , MeanC {agents_res['mean_cost']} EUR"
        f", MeanSC {agents_res['mean_sc']}, MeanSS {agents_res['mean_ss']}"
    )


def get_agents_res(agents_file):
    res_dict = {}

    with open(agents_file, "r") as f:
        data = json.load(f)
        data_res = {"price": {}, "p_res": {}, "p_cons": {}, "p_gen": {}}
        days_plot_data = {"price": {}, "p_res": {}, "p_cons": {}, "p_gen": {}}

        # go by all agents and sort the results in lists
        for agent, results in data.items():
            data_res["price"][agent] = []
            data_res["p_res"][agent] = []
            data_res["p_cons"][agent] = []
            data_res["p_gen"][agent] = []
            days_plot_data["price"][agent] = []
            days_plot_data["p_res"][agent] = []
            days_plot_data["p_cons"][agent] = []
            days_plot_data["p_gen"][agent] = []
            for idx_day, timestamp in enumerate(results.keys()):
                data_res["price"][agent].extend(results[timestamp]["price"])
                data_res["p_res"][agent].extend(results[timestamp]["p_res"])
                data_res["p_cons"][agent].extend(results[timestamp]["p_cons"])
                data_res["p_gen"][agent].extend(results[timestamp]["p_gen"])
        res_dict["df"] = pd.DataFrame(data_res["p_res"])

        # amount of energy and costs
        amount_e = []
        amount_e_feedin = []
        amount_e_demand = []
        amount_e_gen = []
        amount_e_cons = []
        amount_costs = []
        for name, profile in data_res["p_res"].items():
            amount_e.append(round(sum(profile) / 4))
            amount_e_feedin.append(round(sum(val for val in profile if val < 0) / 4))
            amount_e_demand.append(round(sum(val for val in profile if val > 0) / 4))
            amount_e_cons.append(round(sum(data_res["p_cons"][name]) / 4))
            amount_e_gen.append(round(sum(data_res["p_gen"][name]) / 4))
            if "cost_sum" in data_res:
                amount_costs.append(round(sum(data_res["cost_sum"][name])))
            else:
                grid_cost = (
                    sum(
                        [
                            p * c
                            for p, c in zip(
                                data_res["p_res"][name], data_res["price"][name]
                            )
                            if p > 0
                        ]
                    )
                    / 4
                )
                grid_remun = (
                    -sum([p * 0.07 for p in data_res["p_res"][name] if p < 0]) / 4
                )
                amount_costs.append(round(grid_cost - grid_remun))

        # self-consumption and self-sufficiency
        sc_ratio = []
        ss_ratio = []
        for idx, name in enumerate(data_res["p_res"].keys()):
            if amount_e_gen[idx] == 0:
                sc_ratio.append(0)
            else:
                sc_ratio.append(
                    round(
                        (amount_e_gen[idx] - amount_e_feedin[idx])
                        / amount_e_gen[idx]
                        * 100,
                        4,
                    )
                )
            ss_ratio.append(
                round(100 - (amount_e_demand[idx] / amount_e_cons[idx]) * 100, 4)
            )

        res_dict["amount_e_list"] = amount_e
        res_dict["amount_efeedin_list"] = amount_e_feedin
        res_dict["amount_edemand_list"] = amount_e_demand
        res_dict["amount_e_cons"] = amount_e_cons
        res_dict["amount_e_gen"] = amount_e_gen
        res_dict["mean_energy"] = round(sum(amount_e) / len(amount_e), 2)
        res_dict["mean_cost"] = round(sum(amount_costs) / len(amount_costs), 2)
        res_dict["mean_sc"] = round(sum(sc_ratio) / len(sc_ratio), 4)
        res_dict["mean_ss"] = round(sum(ss_ratio) / len(ss_ratio), 4)

    return res_dict


def main():
    day = 0

    if len(sys.argv) > 1:
        rundir = sys.argv[1]
    else:
        print("Need to specify the directory of the run to plot.")
        return

    print("########################################")
    print(f"## Results for simulation {rundir} ##")
    print("########################################")

    if len(sys.argv) == 2:
        print("No plotting for days, just overall results")
        plot_sim_run(rundir)

    if len(sys.argv) == 3:
        day = int(sys.argv[2])
        print(f"Plotting for day: {day}")
        plot_sim_run(rundir, [day])

    if len(sys.argv) > 3:
        day_start = int(sys.argv[2])
        day_end = int(sys.argv[3])
        days_plot = list(range(day_start, day_end + 1))
        print(f"Plotting for days: {days_plot}")
        plot_sim_run(rundir, days_plot)


if __name__ == "__main__":
    main()
    # to run, use something like python src/plot_things.py "outputs/0" 0
