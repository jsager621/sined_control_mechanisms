import asyncio
import logging
from mango import create_container
from central_instance import CentralInstance
from participant import NetParticipant
from syncing_agent import SyncingAgent
from messages.message_classes import get_codec
from datetime import datetime
from util import read_simulation_config, read_grid_config

HOST = "localhost"
PORT = 5555

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


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


def process_outputs(agents):
    pass


async def main():
    sim_config = read_simulation_config()

    str_start = sim_config["start_time"]
    str_end = sim_config["end_time"]
    unix_start = datetime.fromisoformat(str_start).timestamp()
    unix_end = datetime.fromisoformat(str_end).timestamp()

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
    agents = participants.append(central_instance)
    process_outputs(agents)


if __name__ == "__main__":
    asyncio.run(main())
