import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_results import get_line_load, get_vm_pu, get_agents_res

# import matplotlib.dates as mdates
# import copy


def timesteps_to_datetime(timesteps_np: np.ndarray) -> np.datetime64:
    time_delta = np.timedelta64(900, "s") * timesteps_np
    # calculate the datetime corresponding to `timestep`
    datetimes = np.datetime64("2020-01-01 00:00:00") + time_delta
    return datetimes


def comp_results_line(res_dict):
    # range from max to min value for all simulations
    list_max_val = []
    list_min_val = []
    for name, res_sim in res_dict.items():
        list_max_val.append(res_sim["max_val"])
        list_min_val.append(res_sim["min_val"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_max_val, bottom=list_min_val, width=0.8)
    for i, value in enumerate(list_max_val):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.text(i, list_min_val[i] - 1, str(list_min_val[i]), ha="center", va="top")
    plt.ylabel("value range for line loadings, in %")
    plt.ylim((0, max(1.1 * max(list_max_val), 100)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "line_val_range.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # number of violations plus overload work
    list_violations = []
    list_ovl_work = []
    for name, res_sim in res_dict.items():
        list_violations.append(res_sim["num_ovl_sum"])
        list_ovl_work.append(res_sim["ovl_work_kWh"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_violations, width=0.8)
    for i, value in enumerate(list_violations):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("number of violated steps")
    plt.ylim((0, max(1.1 * max(list_violations), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "line_num_viol.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_ovl_work, width=0.8)
    for i, value in enumerate(list_ovl_work):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("overload work, in kWh")
    plt.ylim((0, max(1.1 * max(list_ovl_work), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "line_ovl_work.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )


def comp_results_bus(res_dict):
    # range from max to min value for all simulations
    list_max_val = []
    list_min_val = []
    for name, res_sim in res_dict.items():
        list_max_val.append(res_sim["max_val"])
        list_min_val.append(res_sim["min_val"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_max_val, bottom=list_min_val, width=0.8)
    for i, value in enumerate(list_max_val):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
        plt.text(i, list_min_val[i] - 1, str(list_min_val[i]), ha="center", va="top")
    plt.ylabel("value range for bus voltage magnitude, in p.u.")
    plt.ylim((0, max(1.1 * max(list_max_val), 100)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "bus_val_range.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # number of violations plus overload work
    list_vio_up = []
    list_vio_low = []
    list_ovl_sum_up = []
    list_ovl_sum_lw = []
    for name, res_sim in res_dict.items():
        list_vio_up.append(sum(res_sim["num_ovl_up_list"]))
        list_vio_low.append(sum(res_sim["num_ovl_lw_list"]))
        list_ovl_sum_up.append(res_sim["ovl_sum_ov"])
        list_ovl_sum_lw.append(res_sim["ovl_sum_lw"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_vio_up, width=0.8)
    plt.bar(res_dict.keys(), list_vio_low, width=0.8)
    for i, value in enumerate(list_vio_up):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(list_vio_low):
        plt.text(i, value - 1, str(value), ha="center", va="top")
    plt.ylabel("number of violated steps")
    plt.ylim((min(1.1 * min(list_vio_low), -1), max(1.1 * max(list_vio_up), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "bus_num_viol.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_ovl_sum_up, width=0.8)
    plt.bar(res_dict.keys(), list_ovl_sum_lw, width=0.8)
    for i, value in enumerate(list_ovl_sum_up):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(list_ovl_sum_lw):
        plt.text(i, value - 1, str(value), ha="center", va="top")
    plt.ylabel("bus sum voltage limit violation, in p.u.")
    plt.ylim((min(1.1 * min(list_ovl_sum_lw), -1), max(1.1 * max(list_ovl_sum_up), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "bus_vm_ovl_sum.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )


def comp_results_agents(res_dict):
    # energy an cost values
    list_energy = []
    list_cost = []
    list_sc = []
    list_ss = []
    for name, res_sim in res_dict.items():
        list_energy.append(res_sim["mean_energy"])
        list_cost.append(res_sim["mean_cost"])
        list_sc.append(res_sim["mean_sc"])
        list_ss.append(res_sim["mean_ss"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_energy, width=0.8)
    for i, value in enumerate(list_energy):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("average energy, in kWh")
    plt.ylim((0, max(1.1 * max(list_energy), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "agent_energy_avg.png"), dpi=300, format="png"
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_cost, width=0.8)
    for i, value in enumerate(list_cost):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("average costs, in EUR")
    plt.ylim((0, max(1.1 * max(list_cost), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "agent_cost_avg.png"), dpi=300, format="png"
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_sc, width=0.8)
    for i, value in enumerate(list_sc):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("average self-consumption, in %")
    plt.ylim((0, max(1.1 * max(list_sc), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "agent_sc_avg.png"), dpi=300, format="png"
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_ss, width=0.8)
    for i, value in enumerate(list_ss):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("average self-sufficiency, in %")
    plt.ylim((0, max(1.1 * max(list_ss), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.savefig(
        os.path.join("outputs", "comp", "agent_ss_avg.png"), dpi=300, format="png"
    )


def main():
    day = 0

    if len(sys.argv) == 1:
        print("Need to specify the simulation results to compare.")
        return

    dir_path = os.path.join("outputs", "comp")
    # delete directory if already existent
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        os.rmdir(dir_path)  # shutil.rmtree(directory_path)
    # create directory anew
    os.makedirs(dir_path)

    print("#########################################")
    print("### Comparing results for simulations ###")
    print("#########################################")

    sim_res = {"bus": {}, "line": {}, "agents": {}}

    for idx_sim in range(1, len(sys.argv)):
        sim_name = sys.argv[idx_sim]
        sim_path = os.path.join("outputs", sim_name)

        if not os.path.isdir(sim_path):
            print(f"No results for simulation in {sim_path}")

        print(f"Load Sim Results for {sim_name}")
        sim_res["bus"][sim_name] = get_vm_pu(os.path.join(sim_path, "bus_vm_pu.json"))
        sim_res["line"][sim_name] = get_line_load(
            os.path.join(sim_path, "line_load.json")
        )
        sim_res["agents"][sim_name] = get_agents_res(
            os.path.join(sim_path, "agents.json")
        )

    print(f"Compare Sim Results for Buses: ...")
    comp_results_bus(sim_res["bus"])
    print(f"Done!")

    print(f"Compare Sim Results for Lines: ...")
    comp_results_line(sim_res["line"])
    print(f"Done!")

    print(f"Compare Sim Results for Agents: ...")
    comp_results_agents(sim_res["agents"])
    print(f"Done!")


if __name__ == "__main__":
    main()
    # to run, use something like python src/plot_things.py "outputs/0" 0
