import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(THIS_DIR, "result_data", "plot_data.json")
PLOT_DIR = os.path.join(THIS_DIR, "plots")

# group data by scenarios using modulo:
SCENARIO_NAMES = [
    "100% PV - 100% BSS",
    "100% PV - 66% BSS",
    "100% PV - 33% BSS",

    "66% PV - 100% BSS",
    "66% PV - 66% BSS",
    "66% PV - 33% BSS",

    "33% PV - 100% BSS",
    "33% PV - 66% BSS",
    "33% PV - 33% BSS"]
MECHANISMS = [
    "None",
    "Tariff",
    "Limits",
    "Peak Price",
    "Conditional Power 8kW",
    "Conditional Power 4kW",
    "Conditional Power 6kW",
    "Conditional Power 10kW"
]

def read_data():
    with open(DATA_FILE) as f:
        d = json.load(f)

    return d

def i_to_mechanism(i):
    return MECHANISMS[i // len(SCENARIO_NAMES)]

def i_to_scenario(i):
    return SCENARIO_NAMES[i % len(SCENARIO_NAMES)]

def make_v_step_plot(i_low, i_high, v_step_up, v_step_low, order=False):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=1.4)
    # parse data around a bit:
    # build a table with columns:
    # SCENARIO, MECHANISM, v_step_up, v_step_low
    data = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": []
    }

    for i in range(i_low, i_high):
        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))
        data["v_step_up"].append(v_step_up[i])
        data["v_step_low"].append(v_step_low[i])

    df = pd.DataFrame(
        data=data
    )

    # df.plot(kind='bar', stacked=True)
    if order:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="v_step_up",
        hue="scenario",
        order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )

        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="v_step_low",
            hue="scenario",
            legend=False,
            order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )

        sns.move_legend(g, "upper left")
    else:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="v_step_up",
        hue="scenario"
        )

        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="v_step_low",
            hue="scenario",
            legend=False
        )

        sns.move_legend(g, "upper center")

    for i in g.containers:
        g.bar_label(i,)

    for i in range(df["mechanism"].nunique()):
        plt.axvline(i + 0.5, g.get_ylim()[0], g.get_ylim()[1])

    # g.despine(left=True)
    g.set_xlabel("Mechanism")
    g.set_ylabel("Number of voltage violations")
    # g.set_axis_labels("Mechanism", "Number of voltage violations")
    # g.legend.set_title("")

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"voltage_steps_{i_low}_{i_high}.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()

def make_v_mag_plot(i_low, i_high, v_mag_up, v_mag_low, order=False):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=1.4)
    # parse data around a bit:
    # build a table with columns:
    # SCENARIO, MECHANISM, v_mag_up, v_mag_low
    data = {
        "scenario": [],
        "mechanism": [],
        "v_mag_up": [],
        "v_mag_low": []
    }

    for i in range(i_low, i_high):
        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))

        data["v_mag_up"].append(round(v_mag_up[i], 2))
        data["v_mag_low"].append(round(-1 * v_mag_low[i], 2))

    df = pd.DataFrame(
        data=data
    )

    # df.plot(kind='bar', stacked=True)

    if order:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="v_mag_up",
        hue="scenario",
        order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )

        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="v_mag_low",
            hue="scenario",
            legend=False,
            order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )
    else:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="v_mag_up",
        hue="scenario"
        )

        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="v_mag_low",
            hue="scenario",
            legend=False
        )

    

    for i in range(df["mechanism"].nunique()):
        plt.axvline(i + 0.5, g.get_ylim()[0], g.get_ylim()[1])

    # g.despine(left=True)
    g.set_xlabel("Mechanism")
    g.set_ylabel("Voltage violation sum magnitude in p.u.")
    sns.move_legend(g, "upper center")
    # g.set_axis_labels("Mechanism", "Number of voltage violations")
    # g.legend.set_title("")

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"voltage_mag_{i_low}_{i_high}.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()

def make_line_step_plot(i_low, i_high, line_vio_step, order=False):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=1.4)
    # parse data around a bit:
    # build a table with columns:
    # SCENARIO, MECHANISM, line_vio_step
    data = {
        "scenario": [],
        "mechanism": [],
        "line_vio_step": []
    }

    for i in range(i_low, i_high):
        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))
        data["line_vio_step"].append(line_vio_step[i])

    df = pd.DataFrame(
        data=data
    )

    # df.plot(kind='bar', stacked=True)
    if order:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="line_vio_step",
        hue="scenario",
        order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )
        sns.move_legend(g, "upper left")
    else:
        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="line_vio_step",
            hue="scenario"
        )
        sns.move_legend(g, "upper center")

    for i in g.containers:
        g.bar_label(i,)

    for i in range(df["mechanism"].nunique()):
        plt.axvline(i + 0.5, g.get_ylim()[0], g.get_ylim()[1])

    # g.despine(left=True)
    g.set_xlabel("Mechanism")
    g.set_ylabel("Number of line limit violations")
    
    # g.set_axis_labels("Mechanism", "Number of voltage violations")
    # g.legend.set_title("")

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"line_steps_{i_low}_{i_high}.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()

def make_line_mag_plot(i_low, i_high, line_vio_mag, order=False):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=1.4)
    # parse data around a bit:
    # build a table with columns:
    # SCENARIO, MECHANISM, line_vio_mag
    data = {
        "scenario": [],
        "mechanism": [],
        "line_vio_mag": []
    }

    for i in range(i_low, i_high):
        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))
        data["line_vio_mag"].append(line_vio_mag[i])

    df = pd.DataFrame(
        data=data
    )

    # df.plot(kind='bar', stacked=True)

    # order parameter to hard code weird order from conditional power
    if order:
        g = sns.barplot(
        data=df,
        #kind="bar",
        x="mechanism",
        y="line_vio_mag",
        hue="scenario",
        order=["Conditional Power 4kW", "Conditional Power 6kW", "Conditional Power 8kW", "Conditional Power 10kW"]
        )
    else:
        g = sns.barplot(
            data=df,
            #kind="bar",
            x="mechanism",
            y="line_vio_mag",
            hue="scenario"
        )

    # for i in g.containers:
    #     g.bar_label(i,)

    for i in range(df["mechanism"].nunique()):
        plt.axvline(i + 0.5, g.get_ylim()[0], g.get_ylim()[1])

    # g.despine(left=True)
    g.set_xlabel("Mechanism")
    g.set_ylabel("Line limit overload work in kWh")
    sns.move_legend(g, "upper center")
    # g.set_axis_labels("Mechanism", "Number of voltage violations")
    # g.legend.set_title("")

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"line_mag_{i_low}_{i_high}.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()

def main():
    data = read_data()

    v_step_up = data["v_step_up"]
    v_step_low = data["v_step_low"]
    v_mag_up = data["v_mag_up"]
    v_mag_low = data["v_mag_low"]

    line_vio_step = data["line_vio_step"]
    line_vio_mag = data["line_vio_mag"]

    # initial results plot
    i_low = 0
    i_high = 45
    make_line_mag_plot(i_low, i_high, line_vio_mag)
    make_line_step_plot(i_low, i_high, line_vio_step)
    make_v_step_plot(i_low, i_high, v_step_up, v_step_low)
    make_v_mag_plot(i_low, i_high, v_mag_up, v_mag_low)
    
    # conditional power comparison plot
    i_low = 36
    i_high = 72
    make_v_step_plot(i_low, i_high, v_step_up, v_step_low, order=True)
    make_v_mag_plot(i_low, i_high, v_mag_up, v_mag_low, order=True)
    make_line_step_plot(i_low, i_high, line_vio_step, order=True)
    make_line_mag_plot(i_low, i_high, line_vio_mag, order=True)



if __name__ == "__main__":
    main()