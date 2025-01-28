import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(THIS_DIR, "result_data", "plot_data.json")
PLOT_DIR = os.path.join(THIS_DIR, "plots")

# group data by scenarios using modulo:
SCENARIO_NAMES = [
    "0",
    "1",
    "2",

    "3",
    "4",
    "5",

    "6",
    "7",
    "8"]
MECHANISMS = [
    "None",
    "Tariff",
    "Limits",
    "Peak",
    "2kW",
    "4kW",
    "Cond",
    "6kW - 10ct/kWh",
    "8kW",
    "10kW"
]

def read_data():
    with open(DATA_FILE) as f:
        d = json.load(f)

    return d

def i_to_mechanism(i):
    return MECHANISMS[i // len(SCENARIO_NAMES)]

def i_to_scenario(i):
    return SCENARIO_NAMES[i % len(SCENARIO_NAMES)]

def make_main_run_plots(raw_data):
    v_step_up = raw_data["v_step_up"]
    v_step_low = raw_data["v_step_low"]
    v_mag_up = raw_data["v_mag_up"]
    v_mag_low = raw_data["v_mag_low"]
    line_vio_step = raw_data["line_vio_step"]
    line_vio_mag = raw_data["line_vio_mag"]

    P12_SCENARIOS = [
            "0",
            "1",
            "2",
            "3"
    ]

    P34_SCENARIOS = [
        "4",
        "5",
        "6",
        "7",
        "8"
    ]

    MAIN_MECHANISMS = [
        "None",
        "Tariff",
        "Peak",
        "Cond",
    ]
    
    
    p1_data = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    for i in range(len(v_step_up)):
        if i_to_scenario(i) not in P12_SCENARIOS:
            continue
        if i_to_mechanism(i) not in MAIN_MECHANISMS:
            continue

        p1_data["scenario"].append(i_to_scenario(i))
        p1_data["mechanism"].append(i_to_mechanism(i))
        p1_data["v_step_up"].append(v_step_up[i])
        p1_data["v_step_low"].append(v_step_low[i])
        p1_data["line_vio_step"].append(line_vio_step[i])
        p1_data["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")

    p1_df = pd.DataFrame(
        data=p1_data
    )

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=2.5)

    fig, ax = plt.subplots()

    g = sns.scatterplot(
        data=p1_df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_up",
        s=600,
        markers=['^', '^', '^', '^'],
        style="mechanism",
        hue="mechanism",
        ax=ax,
        legend=False
        )
    ax.set(xlabel='scenario and mechanism', ylabel="number of violated time steps")

    x_tick_labels = [f"{p1_data["mechanism"][i]}\n{p1_data["scenario"][i]}" for i in range(len(p1_data["mechanism"]))]
    for i in range(len(x_tick_labels)):
        if i % 4 == 1:
            x_tick_labels[i] = f"{p1_data["mechanism"][i]}\n{p1_data["scenario"][i]}"
        else:
            x_tick_labels[i] = f"\n{p1_data["scenario"][i]}"

    g.set_xticklabels(x_tick_labels)
    g = sns.scatterplot(
        data=p1_df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_low",
        s=600,
        markers=['v', 'v', 'v', 'v'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )
    g = sns.scatterplot(
        data=p1_df,
        #kind="bar",
        x="scenario_mechanism",
        y="line_vio_step",
        s=600,
        markers=['s', 's', 's', 's'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )

    for i in range(p1_df["mechanism"].nunique()-1):
        plt.axvline(i*4 + 3.5, g.get_ylim()[0], g.get_ylim()[1])

    plt.legend(title='Violated Timesteps', loc='upper right', labels=['upper voltage band', 'lower voltage band', 'line power'])
    leg = ax.get_legend()
    leg.legend_handles[0].set_facecolor('black')
    leg.legend_handles[0].set_edgecolor('black')
    leg.legend_handles[1].set_facecolor('black')
    leg.legend_handles[1].set_edgecolor('black')
    leg.legend_handles[2].set_facecolor('black')
    leg.legend_handles[2].set_edgecolor('black')
    # plt.show()

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"violated_steps_1.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()

    #------------------------------------------
    #------------------------------------------
    #------------------------------------------
    #------------------------------------------
    #------------------------------------------
    p2_data = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    for i in range(len(v_step_up)):
        if i_to_scenario(i) not in P34_SCENARIOS:
            continue
        if i_to_mechanism(i) not in MAIN_MECHANISMS:
            continue

        p2_data["scenario"].append(i_to_scenario(i))
        p2_data["mechanism"].append(i_to_mechanism(i))
        p2_data["v_step_up"].append(v_step_up[i])
        p2_data["v_step_low"].append(v_step_low[i])
        p2_data["line_vio_step"].append(line_vio_step[i])
        p2_data["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")

    p2_df = pd.DataFrame(
        data=p2_data
    )

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=2.5)

    fig, ax = plt.subplots()

    g = sns.scatterplot(
        data=p2_df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_up",
        s=600,
        markers=['^', '^', '^', '^'],
        style="mechanism",
        hue="mechanism",
        ax=ax,
        legend=False
        )
    ax.set(xlabel='scenario and mechanism', ylabel="number of violated time steps")

    x_tick_labels = [f"{p2_data["mechanism"][i]}\n{p2_data["scenario"][i]}" for i in range(len(p2_data["mechanism"]))]
    for i in range(len(x_tick_labels)):
        if i % 5 == 2:
            x_tick_labels[i] = f"{p2_data["mechanism"][i]}\n{p2_data["scenario"][i]}"
        else:
            x_tick_labels[i] = f"\n{p2_data["scenario"][i]}"

    g.set_xticklabels(x_tick_labels)
    g = sns.scatterplot(
        data=p2_df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_low",
        s=600,
        markers=['v', 'v', 'v', 'v'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )
    g = sns.scatterplot(
        data=p2_df,
        #kind="bar",
        x="scenario_mechanism",
        y="line_vio_step",
        s=600,
        markers=['s', 's', 's', 's'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )

    for i in range(p2_df["mechanism"].nunique()-1):
        plt.axvline(i*5 + 4.5, g.get_ylim()[0], g.get_ylim()[1])

    plt.legend(title='Violated Timesteps', loc='lower right', labels=['upper voltage band', 'lower voltage band', 'line power'])
    leg = ax.get_legend()
    leg.legend_handles[0].set_facecolor('black')
    leg.legend_handles[0].set_edgecolor('black')
    leg.legend_handles[1].set_facecolor('black')
    leg.legend_handles[1].set_edgecolor('black')
    leg.legend_handles[2].set_facecolor('black')
    leg.legend_handles[2].set_edgecolor('black')
    # plt.show()

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"violated_steps_2.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    plt.cla()



#
#
#
def make_cond_plot_1(raw_data):
    v_step_up = raw_data["v_step_up"]
    v_step_low = raw_data["v_step_low"]
    line_vio_step = raw_data["line_vio_step"]

    RELEVANT_SCENARIOS = [
        "0",
        "1",
        "2",
        "3"
    ]

    COND_MECHANISMS = [
        "2kW",
        "4kW",
        "Cond",
        "6kW - 10ct/kWh",
        "8kW",
        "10kW"
    ]
    
    
    data = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    append_back = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    for i in range(len(v_step_up)):
        if i_to_scenario(i) not in RELEVANT_SCENARIOS:
            continue
        if i_to_mechanism(i) not in COND_MECHANISMS:
            continue

        if i_to_mechanism(i) == "6kW - 10ct/kWh":
            append_back["scenario"].append(i_to_scenario(i))
            append_back["mechanism"].append(i_to_mechanism(i))
            append_back["v_step_up"].append(v_step_up[i])
            append_back["v_step_low"].append(v_step_low[i])
            append_back["line_vio_step"].append(line_vio_step[i])
            append_back["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")
            continue

        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))
        data["v_step_up"].append(v_step_up[i])
        data["v_step_low"].append(v_step_low[i])
        data["line_vio_step"].append(line_vio_step[i])
        data["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")

    for k in append_back.keys():
        data[k] = data[k] + append_back[k]

    df = pd.DataFrame(
        data=data
    )

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=2.5)

    fig, ax = plt.subplots()

    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_up",
        s=600,
        markers=['^'],
        style="mechanism",
        hue="mechanism",
        ax=ax,
        legend=False
        )
    ax.set(xlabel='scenario and mechanism', ylabel="number of violated time steps")

    x_tick_labels = [f"{data["mechanism"][i]}\n{data["scenario"][i]}" for i in range(len(data["mechanism"]))]
    for i in range(len(x_tick_labels)):
        if i % 4 == 1:
            x_tick_labels[i] = f"{data["mechanism"][i]}\n{data["scenario"][i]}"
            if data["mechanism"][i] == "Cond":
                x_tick_labels[i] = f"6kW\n{data["scenario"][i]}"
            if data["mechanism"][i] == "6kW - 10ct/kWh":
                x_tick_labels[i] = f"6kW - 10ct\n{data["scenario"][i]}"
        else:
            x_tick_labels[i] = f"\n{data["scenario"][i]}"
    g.set_xticklabels(x_tick_labels)

    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_low",
        s=600,
        markers=['v'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )
    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="line_vio_step",
        s=600,
        markers=['s'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )

    # for i in range(df["mechanism"].nunique()-1):
    #     plt.axvline(i*5 + 4.5, g.get_ylim()[0], g.get_ylim()[1])

    plt.legend(title='Violated Timesteps', loc='upper left', labels=['upper voltage band', 'lower voltage band', 'line power'])
    leg = ax.get_legend()
    leg.legend_handles[0].set_facecolor('black')
    leg.legend_handles[0].set_edgecolor('black')
    leg.legend_handles[1].set_facecolor('black')
    leg.legend_handles[1].set_edgecolor('black')
    leg.legend_handles[2].set_facecolor('black')
    leg.legend_handles[2].set_edgecolor('black')

    for i in range(df["mechanism"].nunique()-1):
        plt.axvline(i*4 + 3.5, g.get_ylim()[0], g.get_ylim()[1])

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"conditional_power_comparison_1.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # plt.show()
    plt.cla()

def make_cond_plot_2(raw_data):
    v_step_up = raw_data["v_step_up"]
    v_step_low = raw_data["v_step_low"]
    line_vio_step = raw_data["line_vio_step"]

    RELEVANT_SCENARIOS = [
        "4",
        "5",
        "6",
        "7",
        "8"
    ]

    COND_MECHANISMS = [
        "2kW",
        "4kW",
        "Cond",
        "6kW - 10ct/kWh",
        "8kW",
        "10kW"
    ]
    
    
    data = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    append_back = {
        "scenario": [],
        "mechanism": [],
        "v_step_up": [],
        "v_step_low": [],
        "line_vio_step": [],
        "scenario_mechanism": []
    }

    for i in range(len(v_step_up)):
        if i_to_scenario(i) not in RELEVANT_SCENARIOS:
            continue
        if i_to_mechanism(i) not in COND_MECHANISMS:
            continue

        if i_to_mechanism(i) == "6kW - 10ct/kWh":
            append_back["scenario"].append(i_to_scenario(i))
            append_back["mechanism"].append(i_to_mechanism(i))
            append_back["v_step_up"].append(v_step_up[i])
            append_back["v_step_low"].append(v_step_low[i])
            append_back["line_vio_step"].append(line_vio_step[i])
            append_back["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")
            continue

        data["scenario"].append(i_to_scenario(i))
        data["mechanism"].append(i_to_mechanism(i))
        data["v_step_up"].append(v_step_up[i])
        data["v_step_low"].append(v_step_low[i])
        data["line_vio_step"].append(line_vio_step[i])
        data["scenario_mechanism"].append(f"{i_to_scenario(i)}_{i_to_mechanism(i)}")

    for k in append_back.keys():
        data[k] = data[k] + append_back[k]

    df = pd.DataFrame(
        data=data
    )

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set(font_scale=2.5)

    fig, ax = plt.subplots()

    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_up",
        s=600,
        markers=['^'],
        style="mechanism",
        hue="mechanism",
        ax=ax,
        legend=False
        )
    ax.set(xlabel='scenario and mechanism', ylabel="number of violated time steps")

    x_tick_labels = [f"{data["mechanism"][i]}\n{data["scenario"][i]}" for i in range(len(data["mechanism"]))]
    for i in range(len(x_tick_labels)):
        if i % 5 == 2:
            x_tick_labels[i] = f"{data["mechanism"][i]}\n{data["scenario"][i]}"
            if data["mechanism"][i] == "Cond":
                x_tick_labels[i] = f"6kW\n{data["scenario"][i]}"
            if data["mechanism"][i] == "6kW - 10ct/kWh":
                x_tick_labels[i] = f"6kW - 10ct\n{data["scenario"][i]}"
        else:
            x_tick_labels[i] = f"\n{data["scenario"][i]}"
    g.set_xticklabels(x_tick_labels)

    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="v_step_low",
        s=600,
        markers=['v'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )
    g = sns.scatterplot(
        data=df,
        #kind="bar",
        x="scenario_mechanism",
        y="line_vio_step",
        s=600,
        markers=['s'],
        style="mechanism",
        hue="mechanism",
        legend=False,
        ax=ax
        )

    # for i in range(df["mechanism"].nunique()-1):
    #     plt.axvline(i*5 + 4.5, g.get_ylim()[0], g.get_ylim()[1])

    plt.legend(title='Violated Timesteps', loc='lower right', labels=['upper voltage band', 'lower voltage band', 'line power'])
    leg = ax.get_legend()
    leg.legend_handles[0].set_facecolor('black')
    leg.legend_handles[0].set_edgecolor('black')
    leg.legend_handles[1].set_facecolor('black')
    leg.legend_handles[1].set_edgecolor('black')
    leg.legend_handles[2].set_facecolor('black')
    leg.legend_handles[2].set_edgecolor('black')

    for i in range(df["mechanism"].nunique()-1):
        plt.axvline(i*5 + 4.5, g.get_ylim()[0], g.get_ylim()[1])

    g.figure.set_size_inches(20,12)

    plt.savefig(
        os.path.join(PLOT_DIR, f"conditional_power_comparison_2.png"),
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    # plt.show()
    plt.cla()

def main():
    data = read_data()
    # make_main_run_plots(data)
    make_cond_plot_1(data)
    make_cond_plot_2(data)


if __name__ == "__main__":
    main()