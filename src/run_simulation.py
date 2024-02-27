import asyncio
from mango import create_container
from central_instance import CentralInstance
from participant import NetParticipant
from syncing_agent import SyncingAgent
from messages.message_classes import get_codec
from datetime import datetime

HOST = "localhost"
PORT = 5555


async def create_agents_and_containers():
    # TODO implement me
    # returns list of agents created via config
    # everything single container for now because no reason not to

    c = await create_container(addr=(HOST, PORT), codec=get_codec())

    agents = [
        NetParticipant(c),
        NetParticipant(c),
        NetParticipant(c),
        CentralInstance(c),
    ]
    agent_addresses = [a.get_address() for a in agents]
    sync_agent = SyncingAgent(c, agent_addresses)
    return (sync_agent, agents, [c])


def process_outputs(agents):
    pass


async def main():
    sync_agent, participants, containers = await create_agents_and_containers()

    str_start = "2020-01-01 01:00:00"
    str_end = "2020-01-01 02:00:00"
    unix_start = datetime.fromisoformat(str_start).timestamp()
    unix_end = datetime.fromisoformat(str_end).timestamp()

    # in current implementation agents are reactive so we only need to run
    # the simulation agent here
    await sync_agent.run(unix_start, unix_end)

    for c in containers:
        await c.shutdown()

    # presumably do some data collecting and saving here
    process_outputs(participants)


if __name__ == "__main__":
    asyncio.run(main())
