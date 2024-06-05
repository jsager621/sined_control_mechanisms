import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import copy


def plot_sim_run(rundir, day_start, day_end):
    # assumed contents of the directory:
    # n times agent<n>.json
    # bus_vm_pu.json
    # line_load.json
    if day_start == day_end:
            days = [day_start]
    else:
        days = list(range(day_start, day_end+1))


    plot_line_load(os.path.join(rundir, "line_load.json"), rundir, days)
    plot_vm_pu(os.path.join(rundir, "bus_vm_pu.json"), rundir, days)
    plot_agents(os.path.join(rundir, "agents.json"), rundir, days)

def plot_line_load(line_load_file, rundir, days):
    with open(line_load_file, "r") as f:
        data = json.load(f)
        plot_data = {}

        # filter day
        for day in days:
            for key in data.keys():
                if day == days[0]:
                    plot_data[key] = data[key][day]
                else:
                    plot_data[key] += data[key][day]

        df = pd.DataFrame(plot_data)
        fig = px.line(df)
        outfile = os.path.join(rundir, "line_load.html")
        fig.write_html(outfile)


def plot_vm_pu(vm_pu_file, rundir, days):
    with open(vm_pu_file, "r") as f:
        data = json.load(f)
        plot_data = {}

        # filter day
        for day in days:
            for key in data.keys():
                if day == days[0]:
                    plot_data[key] = data[key][day]
                else:
                    plot_data[key] += data[key][day]

        df = pd.DataFrame(plot_data)
        fig = px.line(df)
        outfile = os.path.join(rundir, "vm_pu.html")
        fig.write_html(outfile)


def plot_agents(agents_file, rundir, days):
    pass


def main():
    day = 0

    if len(sys.argv) > 1:
        rundir = sys.argv[1]
    else:
        print("Need to specify the directory of the run to plot.")
        return

    if len(sys.argv) == 3:
        day = int(sys.argv[2])
        plot_sim_run(rundir, day, day)

    if len(sys.argv) > 3:
        day_start = int(sys.argv[2])
        day_end = int(sys.argv[3])
        plot_sim_run(rundir, day_start, day_end)

    


if __name__ == "__main__":
    main()
    # to run, use something like python src/plot_things.py "outputs/0" 0
