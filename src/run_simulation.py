import asyncio
import logging
import os
import json
import numpy as np
from mango import create_container
from central_instance import CentralInstance
from participant import NetParticipant
from syncing_agent import SyncingAgent
from messages.message_classes import get_codec
from datetime import datetime
from util import read_simulation_config, read_grid_config, time_str_to_int
import random
import pandas as pd

HOST = "localhost"
PORT = 5556

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


async def create_agents_and_containers(grid_config):
    # returns list of agents created via config
    # everything single container for now because no reason not to
    n_participants = grid_config["NUM_PARTICIPANTS"]

    c = await create_container(addr=(HOST, PORT), codec=get_codec())

    # participant ID decides which devices it has from the config ratios
    # read ratios
    r_pv = grid_config["R_PV"]
    r_ev = grid_config["R_EV"]
    r_bss = grid_config["R_BSS"]
    r_cs = grid_config["R_CS"]
    r_hp = grid_config["R_HP"]

    # make bool lists with ratio amount of true values
    # then shuffle each list
    # devices of agent i are given by the values in list i for each list
    pv = [i < r_pv * n_participants for i in range(n_participants)]
    ev = [i < r_ev * n_participants for i in range(n_participants)]
    bss = [i < r_bss * n_participants for i in range(n_participants)]
    cs = [i < r_cs * n_participants for i in range(n_participants)]
    hp = [i < r_hp * n_participants for i in range(n_participants)]

    random.shuffle(pv)
    random.shuffle(ev)
    random.shuffle(bss)
    random.shuffle(cs)
    random.shuffle(hp)

    participants = []
    for i in range(n_participants):
        participants.append(
            NetParticipant(
                container=c,
                has_pv=pv[i],
                has_ev=ev[i],
                has_bss=pv[i] and bss[i],
                has_cs=cs[i],
                has_hp=hp[i],
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

    central_instance = CentralInstance(c)
    agents = participants + [central_instance]

    agent_addresses = [a.get_address() for a in agents]
    sync_agent = SyncingAgent(c, agent_addresses)
    return (sync_agent, participants, central_instance, [c])


def process_outputs(participants, central_instance):
    OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
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
    sim_config = read_simulation_config()
    random.seed(sim_config["seed"])

    str_start = sim_config["start_time"]
    str_end = sim_config["end_time"]
    unix_start = time_str_to_int(str_start)
    unix_end = time_str_to_int(str_end)

    grid_config = read_grid_config()

    sync_agent, participants, central_instance, containers = (
        await create_agents_and_containers(grid_config)
    )

    for p in participants:
        await p.register_to_central_agent(central_instance.get_address())

    # in current implementation agents are reactive so we only need to run
    # the simulation agent here
    await sync_agent.run(unix_start, unix_end)

    for c in containers:
        await c.shutdown()

    # presumably do some data collecting and saving here
    process_outputs(participants, central_instance)


if __name__ == "__main__":
    asyncio.run(main())
