import os
import shutil
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_results import get_line_load, get_vm_pu, get_agents_res

# import matplotlib.dates as mdates
# import copy

GRAN = 96
DAY_START = 40.75
DAY_END = 40.99
IDX_DAYS_PLOT = [int(GRAN * DAY_START), int(GRAN * DAY_END)]
FORMATs = ["pdf", "png"]
DPI = 500
FONT_SIZE = 18


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
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "line_val_range." + f),
            dpi=DPI,
            format=f,
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
    plt.ylabel("number of violated steps with overload")
    plt.ylim((0, max(1.1 * max(list_violations), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "line_num_viol." + f),
            dpi=DPI,
            format=f,
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
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "line_ovl_work." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_line_profile(res_dict):
    # profile for exemplary day
    fig = plt.figure(figsize=(10, 4))
    for name, res_sim in res_dict.items():
        plt.plot(res_sim["df"]["trafo 1"], label=name)
    plt.xlabel("Simulation time step")
    plt.ylabel("Transformer loading, in %")
    plt.xlim(IDX_DAYS_PLOT[0], IDX_DAYS_PLOT[1])
    plt.ylim(0, 125)
    plt.axhline(y=100, color="r", linestyle="--", linewidth=1)
    plt.legend(loc="upper left")
    fig.get_axes()[0].set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "line_load_profile." + f),
            dpi=DPI,
            format=f,
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
    plt.bar(
        res_dict.keys(),
        [list_max_val[i] - list_min_val[i] for i in range(len(list_max_val))],
        bottom=list_min_val,
        width=0.8,
    )
    for i, value in enumerate(list_max_val):
        plt.text(i, value + 0.01, str(value), ha="center", va="bottom")
        plt.text(i, list_min_val[i] - 0.01, str(list_min_val[i]), ha="center", va="top")
    plt.ylabel("value range for bus voltage magnitude, in p.u.")
    plt.ylim((min(0.9 * min(list_min_val), 0.99), max(1.1 * max(list_max_val), 1.01)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "bus_val_range." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )

    # number of violations plus overload work
    list_vio_up = []
    list_vio_low = []
    list_ovl_sum_up = []
    list_ovl_sum_lw = []
    for name, res_sim in res_dict.items():
        list_vio_up.append(max(res_sim["num_ovl_up_list"]))
        list_vio_low.append(min(res_sim["num_ovl_lw_list"]))
        list_ovl_sum_up.append(res_sim["ovl_sum_ov"])
        list_ovl_sum_lw.append(res_sim["ovl_sum_lw"])

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_vio_up, width=0.8)
    plt.bar(res_dict.keys(), list_vio_low, width=0.8)
    for i, value in enumerate(list_vio_up):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    for i, value in enumerate(list_vio_low):
        plt.text(i, value - 1, str(value), ha="center", va="top")
    plt.ylabel("number of violated steps with voltage")
    plt.ylim((min(1.1 * min(list_vio_low), -1), max(1.1 * max(list_vio_up), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "bus_num_viol." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_ovl_sum_up, width=0.8)
    plt.bar(res_dict.keys(), [-val for val in list_ovl_sum_lw], width=0.8)
    for i, value in enumerate(list_ovl_sum_up):
        plt.text(i, value + 0.1, str(value), ha="center", va="bottom")
    for i, value in enumerate(list_ovl_sum_lw):
        plt.text(i, -value - 0.1, str(-value), ha="center", va="top")
    plt.ylabel("bus sum voltage limit violation, in p.u.")
    plt.ylim((min(-1.1 * max(list_ovl_sum_lw), -1), max(1.1 * max(list_ovl_sum_up), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "bus_vm_ovl_sum." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_bus_profile(res_dict):
    # profile for exemplary day
    fig = plt.figure(figsize=(10, 4.8))
    plt.axhline(y=1.05, color="r", linestyle="--", linewidth=1)
    plt.axhline(y=0.95, color="r", linestyle="--", linewidth=1)
    plt.axhline(y=1, color="grey", linestyle="--", linewidth=1)
    cols = [
        "#005374",
        "#66B4D3",
        "#BE1E3C",
        "#711C2F",
        "#D46700",
        "#6D8300",
        "#4C1830",
        "#B983B2",
        "#00534A",
    ]
    lstls = [
        "-",
        ":",
        "--",
        "-.",
        (5, (10, 3)),
        (0, (3, 5, 1, 5)),
        (0, (1, 5)),
        (0, (5, 5)),
        (0, (1, 1)),
    ]
    for idx, name in enumerate(res_dict.keys()):
        plt.plot(
            res_dict[name]["df"]["KV_3_16"],
            label=name,
            color=cols[idx],
            linestyle=lstls[idx],
        )
    plt.xlabel("Simulation time step")
    plt.ylabel("Bus voltage magnitude, in p.u.")
    plt.xlim(IDX_DAYS_PLOT[0], IDX_DAYS_PLOT[1])
    plt.ylim(0.945, 1.005)
    plt.legend(loc="upper left")
    fig.get_axes()[0].set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "bus_vm_profile." + f),
            dpi=DPI,
            format=f,
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
    plt.ylim((0.98 * min(list_energy), max(1.05 * max(list_energy), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "agent_energy_avg." + f),
            dpi=DPI,
            format=f,
        )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_cost, width=0.8)
    for i, value in enumerate(list_cost):
        plt.text(i, value + 1, str(value), ha="center", va="bottom")
    plt.ylabel("average costs, in EUR")
    plt.ylim((0.98 * min(list_cost), max(1.05 * max(list_cost), 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "agent_cost_avg." + f),
            dpi=DPI,
            format=f,
        )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_sc, width=0.8)
    for i, value in enumerate(list_sc):
        plt.text(i, value + 0.02, str(value), ha="center", va="bottom")
    plt.ylabel("average self-consumption, in %")
    plt.ylim((min(list_sc) - 0.1, max(max(list_sc) + 0.3, 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "agent_sc_avg." + f),
            dpi=DPI,
            format=f,
        )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 4))
    plt.bar(res_dict.keys(), list_ss, width=0.8)
    for i, value in enumerate(list_ss):
        plt.text(i, value + 0.02, str(value), ha="center", va="bottom")
    plt.ylabel("average self-sufficiency, in %")
    plt.ylim((min(list_ss) - 0.1, max(max(list_ss) + 0.3, 1)))
    plt.xticks(rotation=45, ha="right")
    ax.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "agent_ss_avg." + f),
            dpi=DPI,
            format=f,
        )


def comp_results_table(sim_res):
    column_names = [
        "Max. Line loading, in %",
        "Min. Line loading, in %",
        "#Ovl loading",
        "Sum overload work, in kWh",
        "Max. Voltage, in p.u.",
        "Min. Voltage, in p.u.",
        "#Ovl voltage upper",
        "#Ovl voltage lower",
        "Sum over-voltage, in p.u.",
        "Sum under-voltage, in p.u.",
        "Avrg. Energy Prosumer, in kWh",
        "Avrg. Cost Prosumer, in EUR",
        "Avrg. Self-consumption Prosumer, in %",
        "Avrg. Self-sufficiency Prosumer, in %",
    ]
    index_names = list(sim_res["bus"].keys())
    df_res = pd.DataFrame(columns=column_names, index=index_names)
    df_res.loc[df_res.columns[0]] = [None for x in range(len(column_names))]

    # insert line results
    for sim_name, res_dict in sim_res["line"].items():
        df_res.at[sim_name, "Max. Line loading, in %"] = res_dict["max_val"]
        df_res.at[sim_name, "Min. Line loading, in %"] = res_dict["min_val"]
        df_res.at[sim_name, "#Ovl loading"] = res_dict["num_ovl_sum"]
        df_res.at[sim_name, "Sum overload work, in kWh"] = res_dict["ovl_work_kWh"]

    # insert bus results
    for sim_name, res_dict in sim_res["bus"].items():
        df_res.at[sim_name, "Max. Voltage, in p.u."] = res_dict["max_val"]
        df_res.at[sim_name, "Min. Voltage, in p.u."] = res_dict["min_val"]
        df_res.at[sim_name, "#Ovl voltage upper"] = max(res_dict["num_ovl_up_list"])
        df_res.at[sim_name, "#Ovl voltage lower"] = min(res_dict["num_ovl_lw_list"])
        df_res.at[sim_name, "Sum over-voltage, in p.u."] = res_dict["ovl_sum_ov"]
        df_res.at[sim_name, "Sum under-voltage, in p.u."] = res_dict["ovl_sum_lw"]

    # insert agent results
    for sim_name, res_dict in sim_res["agents"].items():
        df_res.at[sim_name, "Avrg. Energy Prosumer, in kWh"] = res_dict["mean_energy"]
        df_res.at[sim_name, "Avrg. Cost Prosumer, in EUR"] = res_dict["mean_cost"]
        df_res.at[sim_name, "Avrg. Self-consumption Prosumer, in %"] = res_dict[
            "mean_sc"
        ]
        df_res.at[sim_name, "Avrg. Self-sufficiency Prosumer, in %"] = res_dict[
            "mean_ss"
        ]

    df_res.to_excel(os.path.join("outputs", "comp", "table.xlsx"))


def comp_results_plots_violations(sim_res):
    # plot for comparison of number of violations
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.grid()
    x_coord = np.arange(len(sim_res["bus"].keys()))
    # voltage values
    num_v = [max(res_dict["num_ovl_up_list"]) for res_dict in sim_res["bus"].values()]
    col_up = (102 / 255, 180 / 255, 211 / 255)
    col_lw = (0 / 255, 83 / 255, 116 / 255)
    ax1.scatter(
        x_coord, num_v, label="Voltage, upper lim.", color=col_up, marker="^", s=120
    )
    num_v = [-min(res_dict["num_ovl_lw_list"]) for res_dict in sim_res["bus"].values()]
    ax1.scatter(
        x_coord, num_v, label="Voltage, lower lim.", color=col_lw, marker="v", s=120
    )
    # loading values
    num_v = [res_dict["num_ovl_sum"] for res_dict in sim_res["line"].values()]
    ax1.scatter(
        x_coord, num_v, label="Loading", color=(109 / 255, 131 / 255, 0 / 255), s=100
    )
    # plt.ylim((min(list_ss) - 0.1, max(max(list_ss) + 0.3, 1)))
    ax1.set_xlabel("")
    ax1.set_ylabel("Number of violations")
    ax1.tick_params(axis="y")
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(sim_res["bus"].keys(), rotation=45, ha="right")
    ax1.legend(loc="upper right")
    # ax1.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_num." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )

    # plot for comparison of aggregated violation
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.grid()
    x_coord = np.arange(len(sim_res["bus"].keys()))
    # first axis with voltage values
    col = (102 / 255, 180 / 255, 211 / 255)
    col_ax = (0 / 255, 83 / 255, 116 / 255)
    num_v = [res_dict["ovl_sum_ov"] for res_dict in sim_res["bus"].values()]
    ax1.scatter(
        x_coord, num_v, label="Voltage, upper lim.", color=col, marker="^", s=120
    )
    num_v = [res_dict["ovl_sum_lw"] for res_dict in sim_res["bus"].values()]
    ax1.scatter(
        x_coord,
        num_v,
        label="Voltage, lower lim.",
        color=col_ax,
        marker="v",
        s=120,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Aggregated voltage violations, in p.u.", color=col_ax)
    ax1.tick_params(axis="y", labelcolor=col_ax)
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(sim_res["bus"].keys(), rotation=45, ha="right")
    # seconds axis with trafo loading
    col = (109 / 255, 131 / 255, 0 / 255)
    ax2 = ax1.twinx()
    num_v = [res_dict["ovl_work_kWh"] / 1000 for res_dict in sim_res["line"].values()]
    ax2.scatter(x_coord, num_v, label="Loading", color=col, s=100)
    ax2.set_ylabel("Aggregated line overload, in MWh", color=col)
    ax2.tick_params(axis="y", labelcolor=col)
    # adds
    ax1.legend(loc="upper right")
    # fig.get_axes()[0].set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_viol." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_plots_extrema(sim_res):
    # plot for comparison of max/min values
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.grid()
    x_coord = np.arange(len(sim_res["bus"].keys()))
    # first axis with voltage values
    col = (102 / 255, 180 / 255, 211 / 255)
    max_v = [res_dict["max_val"] for res_dict in sim_res["bus"].values()]
    min_v = [res_dict["min_val"] for res_dict in sim_res["bus"].values()]
    ax1.bar(x_coord, np.array(max_v) - np.array(min_v), bottom=min_v, color=col)
    ax1.set_xlabel("")
    ax1.set_ylabel("Range for bus voltage magnitude, in p.u.", color=col)
    ax1.tick_params(axis="y", labelcolor=col)
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(sim_res["bus"].keys(), rotation=45, ha="right")
    # seconds axis with trafo loading
    col = (109 / 255, 131 / 255, 0 / 255)
    ax2 = ax1.twinx()
    ax2.scatter(
        x_coord,
        [res_dict["max_val"] for res_dict in sim_res["line"].values()],
        color=col,
    )
    ax2.set_ylabel("Maximum transformer loading, in %", color=col)
    ax2.tick_params(axis="y", labelcolor=col)
    # adds
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_extr." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_plots_agents(sim_res):
    # plot for comparison of costs and self-consumption (compared to reference case)
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.grid()
    x_names_w_ref = list(sim_res["bus"].keys())
    x_ref = x_names_w_ref[0]
    x_names = x_names_w_ref[1:]
    x_coord = np.arange(len(x_names))
    # first axis with costs
    col = (102 / 255, 180 / 255, 211 / 255)
    num_v = [
        (
            sim_res["agents"][res_name]["mean_cost"]
            - sim_res["agents"][x_ref]["mean_cost"]
        )
        for res_name in x_names
    ]
    ax1.scatter(x_coord, num_v, color=col, s=120)
    ax1.set_xlabel("")
    ax1.set_ylabel("Change in average cost, in EUR", color=col)
    ax1.tick_params(axis="y", labelcolor=col)
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(x_names, rotation=45, ha="right")
    # seconds axis with self-consumption
    col = (109 / 255, 131 / 255, 0 / 255)
    ax2 = ax1.twinx()
    num_v = [
        (sim_res["agents"][res_name]["mean_sc"] - sim_res["agents"][x_ref]["mean_sc"])
        for res_name in x_names
    ]
    ax2.scatter(x_coord, num_v, color=col, s=100)
    ax2.set_ylabel("Change in average self-consumption, in %", color=col)
    ax2.tick_params(axis="y", labelcolor=col)
    fig.get_axes()[0].set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_agents_sc." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )

    # plot for comparison of costs and energy sums (compared to reference case)
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.grid()
    x_names_w_ref = list(sim_res["bus"].keys())
    x_ref = x_names_w_ref[0]
    x_names = x_names_w_ref[1:]
    x_coord = np.arange(len(x_names))
    # first axis with costs
    col = (109 / 255, 131 / 255, 0 / 255)
    num_v = [
        (
            sim_res["agents"][res_name]["mean_cost"]
            - sim_res["agents"][x_ref]["mean_cost"]
        )
        for res_name in x_names
    ]
    ax1.scatter(x_coord, num_v, color=col, s=120)
    ax1.set_xlabel("")
    ax1.set_ylabel("Change in average cost, in EUR", color=col)
    ax1.tick_params(axis="y", labelcolor=col)
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(x_names, rotation=45, ha="right")
    # seconds axis with energy sums
    col = (102 / 255, 180 / 255, 211 / 255)
    ax2 = ax1.twinx()
    num_v = [
        (
            sum(sim_res["agents"][res_name]["amount_edemand_list"])
            - sum(sim_res["agents"][x_ref]["amount_edemand_list"])
        )
        for res_name in x_names
    ]
    ax2.scatter(x_coord, num_v, label="Energy Demand", marker="^", color=col, s=100)
    num_v = [
        (
            sum(sim_res["agents"][res_name]["amount_efeedin_list"])
            - sum(sim_res["agents"][x_ref]["amount_efeedin_list"])
        )
        for res_name in x_names
    ]
    ax2.scatter(x_coord, num_v, label="Energy Feedin", marker="v", color=col, s=100)
    ax2.set_ylabel("Change in aggregated energy, in %", color=col)
    ax2.tick_params(axis="y", labelcolor=col)
    fig.get_axes()[0].set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_agents_energy." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_plots_agents_cost(sim_res):
    # plot for comparison of costs and self-consumption (compared to reference case)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.grid()
    x_names_w_ref = list(sim_res["bus"].keys())
    x_ref = x_names_w_ref[0]
    x_names = x_names_w_ref[1:]
    x_coord = np.arange(len(x_names))
    # first axis with costs
    num_v = [
        (
            (
                sim_res["agents"][res_name]["mean_cost"]
                - sim_res["agents"][x_ref]["mean_cost"]
            )
            * 100
        )
        for res_name in x_names
    ]
    for idx in x_coord:
        ax1.text(
            x=idx,
            y=num_v[idx] + 17,
            s=str(round(num_v[idx])),
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax1.scatter(x_coord, num_v, color="k", s=120)
    ax1.set_xlabel("")
    ax1.set_ylabel("Change in average cost, in ct")
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(x_names, rotation=45, ha="right")
    ax1.set_ylim(top=max(num_v) * 1.15)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_costs." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def comp_results_plots_agents_energy(sim_res):
    # plot for energy sums
    fig, ax1 = plt.subplots(figsize=(18, 4))
    ax1.grid()
    x_coord = np.arange(len(sim_res["agents"].keys()))
    num_v = [sum(res_dict["amount_e_cons"]) for res_dict in sim_res["agents"].values()]
    ax1.scatter(
        x_coord,
        num_v,
        label="Consumption",
        color=(102 / 255, 180 / 255, 211 / 255),
        marker="v",
        s=120,
    )
    for i, value in enumerate(num_v):
        plt.text(i, value + 100, str(value), ha="center", va="bottom")
    num_v = [-sum(res_dict["amount_e_gen"]) for res_dict in sim_res["agents"].values()]
    ax1.scatter(
        x_coord,
        num_v,
        label="Generation",
        color=(0 / 255, 83 / 255, 116 / 255),
        marker="v",
        s=120,
    )
    for i, value in enumerate(num_v):
        plt.text(i, value + 100, str(value), ha="center", va="bottom")
    num_v = [
        -sum(res_dict["amount_efeedin_list"]) for res_dict in sim_res["agents"].values()
    ]
    ax1.scatter(
        x_coord,
        num_v,
        label="Feedin",
        color=(102 / 255, 180 / 255, 211 / 255),
        marker="^",
        s=120,
    )
    for i, value in enumerate(num_v):
        plt.text(i, value + 100, str(value), ha="center", va="bottom")
    num_v = [
        sum(res_dict["amount_edemand_list"]) for res_dict in sim_res["agents"].values()
    ]
    ax1.scatter(
        x_coord,
        num_v,
        label="Grid Consumption",
        color=(0 / 255, 83 / 255, 116 / 255),
        marker="^",
        s=120,
    )
    for i, value in enumerate(num_v):
        plt.text(i, value + 100, str(value), ha="center", va="bottom")
    # plt.ylim((min(list_ss) - 0.1, max(max(list_ss) + 0.3, 1)))
    ax1.set_xlabel("")
    ax1.set_ylabel("Average energy, in kWh")
    ax1.tick_params(axis="y")
    ax1.set_xticks(x_coord)
    ax1.set_xticklabels(sim_res["agents"].keys(), rotation=45, ha="right")
    ax1.legend(loc="upper right")
    # ax1.set_rasterized(True)
    plt.rcParams.update({"font.size": FONT_SIZE, "legend.fontsize": FONT_SIZE - 4})
    for f in FORMATs:
        plt.savefig(
            os.path.join("outputs", "comp", "comp_plot_energy." + f),
            dpi=DPI,
            format=f,
            bbox_inches="tight",
        )


def main():
    day = 0

    if len(sys.argv) == 1:
        print("Need to specify the simulation results to compare.")
        return

    dir_path = os.path.join("outputs", "comp")
    # delete directory if already existent
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            os.rmdir(dir_path)
        except:
            shutil.rmtree(dir_path)
            # os.rmdir(dir_path)
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
    # comp_results_bus(sim_res["bus"])
    if len(sim_res["bus"]) < 10:
        comp_results_bus_profile(sim_res["bus"])
    print(f"Done!")

    print(f"Compare Sim Results for Lines: ...")
    # comp_results_line(sim_res["line"])
    if len(sim_res["line"]) < 10:
        comp_results_bus_profile(sim_res["line"])
    print(f"Done!")

    # print(f"Compare Sim Results for Agents: ...")
    # comp_results_agents(sim_res["agents"])
    # print(f"Done!")

    print(f"Create Overview Table for Sim Results: ...")
    comp_results_table(sim_res)
    print(f"Done!")

    print(f"Create Comparing Plots for Sim Results: ...")
    comp_results_plots_violations(sim_res)
    comp_results_plots_extrema(sim_res)
    comp_results_plots_agents(sim_res)
    comp_results_plots_agents_cost(sim_res)
    comp_results_plots_agents_energy(sim_res)
    print(f"Done!")


if __name__ == "__main__":
    main()
    # to run, use something like python src/plot_things.py "outputs/0" 0
