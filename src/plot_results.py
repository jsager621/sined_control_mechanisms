import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy


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
    with open(line_load_file, "r") as f:
        data = json.load(f)
        data_res = {}
        days_plot_data = {}

        # go by all lines and sort the results in lists
        for line, results_list in data.items():
            data_res[line] = []
            days_plot_data[line] = []
            for idx_day, res_list_day in enumerate(results_list):
                for val in res_list_day:
                    data_res[line].append(val)
                    if days is not None and idx_day in days:
                        days_plot_data[line].append(val)
        df_loading = pd.DataFrame(data_res)

        # plot of line loading for the selected day(s)
        if days is not None:
            fig = plt.figure(figsize=(10, 4))
            plt.xlabel("time")
            plt.ylabel("Loading, in %")
            plt.xlim(0, len(days) * 96)
            # default_date_format = mdates.DateFormatter("%m-%d %H:%M")
            # plt.gca().xaxis.set_major_formatter(default_date_format)
            for name, profile in days_plot_data.items():
                plt.plot(profile, label=name)
            # plt.legend()
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
        df_loading.boxplot()
        plt.xticks(rotation=45, ha="right")
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "line_boxpl.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # number of violations
        threshold = 100
        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        n_overloads_list = []
        for name in days_plot_data.keys():
            n_overloads_list.append(sum(df_loading[name] > threshold))
        plt.bar(days_plot_data.keys(), n_overloads_list, width=0.8)
        for i, value in enumerate(n_overloads_list):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.ylabel("number of violated steps")
        plt.ylim((0, max(1.1 * max(n_overloads_list), 1)))
        plt.xticks(rotation=45, ha="right")
        ax.set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "line_num_viol.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # printing of statistics
        max_val = df_loading.max().max()
        min_val = df_loading.min().min()
        mean_val = df_loading.mean().mean()
        num_ovl = sum(n_overloads_list)
        print(f"LINE: Max {max_val} , Min {min_val} , Avg {mean_val} , #Ovl {num_ovl}")


def plot_vm_pu(vm_pu_file, rundir, days):
    with open(vm_pu_file, "r") as f:
        data = json.load(f)
        data_res = {}
        days_plot_data = {}

        # go by all buses and sort the results in lists
        for bus, results_list in data.items():
            data_res[bus] = []
            days_plot_data[bus] = []
            for idx_day, res_list_day in enumerate(results_list):
                for val in res_list_day:
                    data_res[bus].append(val)
                    if days is not None and idx_day in days:
                        days_plot_data[bus].append(val)
        df_voltage = pd.DataFrame(data_res)

        # plot of line loading for the selected day(s)
        if days is not None:
            fig = plt.figure(figsize=(10, 4))
            plt.xlabel("time")
            plt.ylabel("Voltage magnitude, in p.u.")
            plt.xlim(0, len(days) * 96)
            # default_date_format = mdates.DateFormatter("%m-%d %H:%M")
            # plt.gca().xaxis.set_major_formatter(default_date_format)
            for name, profile in days_plot_data.items():
                plt.plot(profile, label=name)
            # plt.legend()
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
        df_voltage.boxplot()
        plt.xticks(rotation=45, ha="right")
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "bus_boxpl.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # number of violations
        threshold_up = 1.05
        threshold_lw = 0.95
        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        n_up_list = []
        n_lw_list = []
        for name in days_plot_data.keys():
            n_up_list.append(sum(df_voltage[name] > threshold_up))
            n_lw_list.append(-sum(df_voltage[name] < threshold_lw))
        plt.bar(days_plot_data.keys(), n_up_list, width=0.8)
        plt.bar(days_plot_data.keys(), n_lw_list, width=0.8)
        for i, value in enumerate(n_up_list):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        for i, value in enumerate(n_lw_list):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.ylabel("number of violated steps")
        plt.ylim((min(1.1 * min(n_lw_list), -1), max(1.1 * max(n_up_list), 1)))
        plt.xticks(rotation=45, ha="right")
        ax.set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "bus_num_viol.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # printing of statistics
        max_val = round(df_voltage.max().max(), 4)
        min_val = round(df_voltage.min().min(), 4)
        mean_val = round(df_voltage.mean().mean(), 4)
        num_up = sum(n_up_list)
        num_lw = sum(n_lw_list)
        print(
            f"BUS: Max {max_val} , Min {min_val} , Avg {mean_val} , #O_up {num_up} , #O_lw {num_lw}"
        )


def plot_agents(agents_file, rundir, days):
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
                if days is not None and idx_day in days:
                    days_plot_data["price"][agent].extend(results[timestamp]["price"])
                    days_plot_data["p_res"][agent].extend(results[timestamp]["p_res"])
                    days_plot_data["p_cons"][agent].extend(results[timestamp]["p_cons"])
                    days_plot_data["p_gen"][agent].extend(results[timestamp]["p_gen"])
        df_p_res = pd.DataFrame(data_res["p_res"])

        # plot of residual power for the selected day(s)
        if days is not None:
            fig = plt.figure(figsize=(10, 4))
            plt.xlabel("time")
            plt.ylabel("Power, in kW")
            plt.xlim(0, len(days) * 96)
            for name, profile in days_plot_data["p_res"].items():
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
        df_p_res.boxplot()
        plt.xticks(rotation=45, ha="right")
        fig.get_axes()[0].set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "agents_boxpl.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # amount of energy and costs
        amount_e = []
        amount_e_feedin = []
        amount_e_demand = []
        amount_e_gen = []
        amount_e_cons = []
        amount_costs = []
        for name, profile in data_res["p_res"].items():
            amount_e.append(round(sum(data_res["p_res"][name]) / 4))
            amount_e_feedin.append(
                round(sum(val for val in data_res["p_res"][name] if val < 0) / 4)
            )
            amount_e_demand.append(
                round(sum(val for val in data_res["p_res"][name] if val > 0) / 4)
            )
            amount_e_cons.append(round(sum(data_res["p_cons"][name]) / 4))
            amount_e_gen.append(round(sum(data_res["p_gen"][name]) / 4))
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
            grid_remun = -sum([p * 0.07 for p in data_res["p_res"][name] if p < 0]) / 4
            amount_costs.append(round(grid_cost - grid_remun))

        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        plt.bar(data_res["p_res"].keys(), amount_e, width=0.8)
        for i, value in enumerate(amount_e):
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

        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        plt.bar(data_res["p_res"].keys(), amount_costs, width=0.8)
        for i, value in enumerate(amount_costs):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.ylabel("Kosten, in EUR")
        plt.ylim((0, max(1.1 * max(amount_costs), 1)))
        plt.xticks(rotation=45, ha="right")
        ax.set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "agents_costs.png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )

        # self-consumption and self-sufficiency
        sc_ratio = []
        ss_ratio = []
        for idx, name in enumerate(data_res["p_res"].keys()):
            if amount_e_gen[idx] == 0:
                sc_ratio.append(0)
            else:
                sc_ratio.append(
                    (amount_e_gen[idx] - amount_e_feedin[idx]) / amount_e_gen[idx] * 100
                )
            ss_ratio.append(100 - (amount_e_demand[idx] / amount_e_cons[idx]) * 100)
        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        plt.bar(data_res["p_res"].keys(), sc_ratio, width=0.8)
        for i, value in enumerate(sc_ratio):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.ylabel("Self-consumption, in %")
        plt.ylim((0, max(1.1 * max(sc_ratio), 1)))
        plt.xticks(rotation=45, ha="right")
        ax.set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "agents_sc.png"),
            dpi=100,
            format="png",
            bbox_inches="tight",
        )
        fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
        plt.bar(data_res["p_res"].keys(), ss_ratio, width=0.8)
        for i, value in enumerate(ss_ratio):
            plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.ylabel("Self-sufficiency, in %")
        plt.ylim((0, max(1.1 * max(ss_ratio), 1)))
        plt.xticks(rotation=45, ha="right")
        ax.set_rasterized(True)
        plt.savefig(
            os.path.join(rundir, "agents_ss.png"),
            dpi=100,
            format="png",
            bbox_inches="tight",
        )

        # printing of statistics
        mean_energy = round(sum(amount_e) / len(amount_e), 1)
        mean_cost = round(sum(amount_costs) / len(amount_costs), 1)
        mean_sc = round(sum(sc_ratio) / len(sc_ratio), 1)
        mean_ss = round(sum(ss_ratio) / len(ss_ratio), 1)
        print(
            f"AGENTS: MeanE {mean_energy} kWh , MeanC {mean_cost} EUR , MeanSC {mean_sc} "
            f", MeanSS {mean_ss}"
        )


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
