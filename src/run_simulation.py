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

HOST = "localhost"
PORT = 5555

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


async def create_agents_and_containers(n_participants):
    # returns list of agents created via config
    # everything single container for now because no reason not to

    c = await create_container(addr=(HOST, PORT), codec=get_codec())

    participants = [NetParticipant(c) for i in range(n_participants)]
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
    with open(filename, 'w') as f:
        f.write(json.dumps(agents_schedule_log, cls=NumpyEncoder))

    # result_timeseries_bus_vm_pu
    # result_timeseries_line_load
    filename = os.path.join(RUNDIR, "bus_vm_pu" + ".json")
    with open(filename, 'w') as f:
            f.write(json.dumps(central_instance.result_timeseries_bus_vm_pu, cls=NumpyEncoder))

    filename = os.path.join(RUNDIR, "line_load" + ".json")
    with open(filename, 'w') as f:
            f.write(json.dumps(central_instance.result_timeseries_line_load, cls=NumpyEncoder))
    


async def main():
    sim_config = read_simulation_config()

    str_start = sim_config["start_time"]
    str_end = sim_config["end_time"]
    unix_start = time_str_to_int(str_start)
    unix_end = time_str_to_int(str_end)

    grid_config = read_grid_config()
    n_participants = grid_config["NUM_PARTICIPANTS"]

    sync_agent, participants, central_instance, containers = (
        await create_agents_and_containers(n_participants)
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
