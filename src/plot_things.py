import seaborn
import os
import sys
import json

def plot_sim_run(rundir):
    # assumed contents of the directory:
    # n times agent<n>.json
    # bus_vm_pu.json
    # line_load.json
    plot_line_load(os.path.join(rundir, "line_load.json"))
    plot_vm_pu(os.path.join(rundir, "bus_vm_pu.json"))

    for f in [x for x in os.listdir(rundir) if x.startswith("agent")]:
        plot_agent(f)


def plot_line_load(line_load_file):
    with open(line_load_file, "r") as f:
        data = json.load(f)

        for line in data.keys():
            print(data[line])

    pass

def plot_vm_pu(vm_pu_file):
    pass

def plot_agent(agent_file):
    pass

def main():
    if len(sys.argv) > 1:
        rundir = sys.argv[1]

    plot_sim_run(rundir)


if __name__ == "__main__":
    main()