import asyncio
import logging
import os
import sys
import json
import numpy as np
from mango import create_container
from central_instance import CentralInstance
from participant import NetParticipant
from syncing_agent import SyncingAgent
from messages.message_classes import get_codec
from datetime import datetime
from util import read_simulation_config, read_grid_config, time_str_to_int, read_prosumer_config, ETG_BASE_GRID_CONF, ETG_CONF_DIR, ETG_PROSUMER_CONF
import random
import pandas as pd
import math

HOST = "localhost"
PORT = 5557

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


async def create_agents_and_containers(grid_config, prosumer_config):
    # returns list of agents created via config
    # everything single container for now because no reason not to
    n_participants = grid_config["NUM_PARTICIPANTS"]

    c = await create_container(addr=(HOST, PORT), codec=get_codec())

    # participant ID decides which devices it has from the config ratios
    # read ratios
    r_pv = grid_config["R_PV"]
    r_ev = grid_config["R_EV"]
    r_bss = grid_config["R_BSS"]
    r_hp = grid_config["R_HP"]

    # make bool lists with ratio amount of true values
    # then shuffle each list
    # devices of agent i are given by the values in list i for each list
    pv = [i < r_pv * n_participants for i in range(n_participants)]
    n_pv = sum(pv)
    ev = [i < r_ev * n_participants for i in range(n_participants)]
    hp = [i < r_hp * n_participants for i in range(n_participants)]

    # randomly distribute these units
    random.shuffle(pv)
    random.shuffle(ev)
    random.shuffle(hp)

    # assign bss to the households with PV units
    n_bss = math.ceil(r_bss * n_pv)
    bss_order = [i < n_bss for i in range(n_pv)]
    random.shuffle(bss_order)
    
    bss = [False for i in range(n_participants)]

    pv_idx = 0
    for i in range(n_participants):
        if not pv[i]:
            continue

        # this is the pv_idx's pv unit
        if bss_order[pv_idx]:
            bss[i] = True

        pv_idx += 1

        if pv_idx > n_pv - 1:
            break

    # sanity check for debugging, no BSS without PV
    assert not any([bss[i] and not pv[i] for i in range(len(pv))])

    participants = []
    for i in range(n_participants):
        participants.append(
            NetParticipant(
                c,
                pv[i],
                ev[i],
                pv[i] and bss[i],
                ev[i],
                hp[i],
                prosumer_config
            )
        )

    df_part = pd.DataFrame(columns=["PV", "EV", "BSS", "CS", "HP"])
    for idx, part in enumerate(participants):
        df_part.loc[idx] = [
            part.dev["pv"]["power_kWp"],
            part.dev["ev"]["capacity_kWh"],
            part.dev["bss"]["capacity_kWh"],
            part.dev["cs"]["power_kW"],
            part.dev["hp"],
        ]
    # print(df_part)

    central_instance = CentralInstance(c, grid_config)
    agents = participants + [central_instance]

    agent_addresses = [a.get_address() for a in agents]
    sync_agent = SyncingAgent(c, agent_addresses)
    return (sync_agent, participants, central_instance, [c])


def process_outputs(participants, central_instance, sub_dir_name=None):
    OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", sub_dir_name)
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    RUNDIR = ""
    for i in range(100):
        RUNDIR = os.path.join(OUTDIR, str(i))
        if os.path.exists(RUNDIR):
            continue

        break

    os.mkdir(RUNDIR)
    agents_schedule_log = {}
    for p in participants:
        agents_schedule_log[p.aid] = p.schedule_log

    filename = os.path.join(RUNDIR, "agents.json")
    with open(filename, "w") as f:
        f.write(json.dumps(agents_schedule_log, cls=NumpyEncoder))

    # result_timeseries_bus_vm_pu
    # result_timeseries_line_load
    filename = os.path.join(RUNDIR, "bus_vm_pu" + ".json")
    with open(filename, "w") as f:
        f.write(
            json.dumps(central_instance.result_timeseries_bus_vm_pu, cls=NumpyEncoder)
        )

    filename = os.path.join(RUNDIR, "line_load" + ".json")
    with open(filename, "w") as f:
        f.write(
            json.dumps(central_instance.result_timeseries_line_load, cls=NumpyEncoder)
        )


async def main():
    # set grid config from command line as necessary
    if len(sys.argv) > 4:
        ctrl_type = sys.argv[1]
        r_pv = float(sys.argv[2])
        r_bss = float(sys.argv[3])
        cond_kw_threshold = int(sys.argv[3])
    else:
        logging.error("This script requires control type, R_PV, R_BSS and cond_kw_threshold as inputs.")
        return

    # different config from old sim
    prosumer_config = read_prosumer_config(ETG_PROSUMER_CONF)
    grid_config = read_grid_config(ETG_BASE_GRID_CONF)

    # set scenario specific params
    grid_config["CONTROL_TYPE"] = ctrl_type
    grid_config["R_PV"] = r_pv
    grid_config["R_BSS"] = r_bss
    grid_config["COND_POWER_THRESHOLD_kW"] = cond_kw_threshold
    

    sim_config = read_simulation_config()
    random.seed(sim_config["seed"])

    str_start = sim_config["start_time"]
    str_end = sim_config["end_time"]
    unix_start = time_str_to_int(str_start)
    unix_end = time_str_to_int(str_end)

    

    sync_agent, participants, central_instance, containers = (
        await create_agents_and_containers(grid_config, prosumer_config)
    )

    for p in participants:
        await p.register_to_central_agent(central_instance.get_address())

    # in current implementation agents are reactive so we only need to run
    # the simulation agent here
    await sync_agent.run(unix_start, unix_end)

    for c in containers:
        await c.shutdown()

    if ctrl_type is not "conditional_power":
        sub_dir_name = ctrl_type
    else:
        sub_dir_name = f"{ctrl_type}_{cond_kw_threshold}"

    # presumably do some data collecting and saving here
    process_outputs(participants, central_instance, sub_dir_name)


if __name__ == "__main__":
    asyncio.run(main())
