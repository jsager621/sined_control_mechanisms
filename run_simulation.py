import asyncio
from mango import create_container
from src.central_instance import CentralInstance
from src.participant import NetParticipant

async def create_agents_and_containers():
    # TODO implement me
    # returns list of agents created via config
    pass

async def simulation_done():
    # TODO implement me
    # can just be a simple future but as a function here to show the
    # intended flow
    pass

def process_outputs(agents):
    pass


async def main():
    agents, containers = await create_agents_and_containers()
    tasks = [asyncio.create_task(a.run()) for a in agents]

    # run for some time
    await simulation_done()

    for c in containers:
        await c.shutdown()

    # presumably do some data collecting and saving here
    process_outputs(agents)


if __name__ == "__main__":
    asyncio.run(main())