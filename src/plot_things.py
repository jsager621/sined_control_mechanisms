import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px

def plot_sim_run(rundir, day):
    # assumed contents of the directory:
    # n times agent<n>.json
    # bus_vm_pu.json
    # line_load.json
    plot_line_load(os.path.join(rundir, "line_load.json"), rundir, day)
    plot_vm_pu(os.path.join(rundir, "bus_vm_pu.json"), rundir, day)
    plot_agents(os.path.join(rundir, "agents.json"), rundir, day)

def plot_line_load(line_load_file, rundir, day):
    with open(line_load_file, "r") as f:
        data = json.load(f)
        
        # filter day
        for key in data.keys():
            data[key] = data[key][day]

        df = pd.DataFrame(data)
        fig = px.line(df)
        outfile = os.path.join(rundir, "line_load.html")
        fig.write_html(outfile)


def plot_vm_pu(vm_pu_file, rundir, day):
    with open(vm_pu_file, "r") as f:
        data = json.load(f)
        
        # filter day
        for key in data.keys():
            data[key] = data[key][day]

        df = pd.DataFrame(data)
        fig = px.line(df)
        outfile = os.path.join(rundir, "vm_pu.html")
        fig.write_html(outfile)

def plot_agents(agents_file, rundir, day):
    pass

def main():
    day = 0

    if len(sys.argv) > 1:
        rundir = sys.argv[1]
    else:
        print("Need to specify the directory of the run to plot.")
        return
    
    if len(sys.argv) > 2:
        day = int(sys.argv[2])

    plot_sim_run(rundir, day)


if __name__ == "__main__":
    main()