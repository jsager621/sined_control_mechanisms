from mango.agent.core import Agent
from messages.message_classes import TimeStepMessage, TimeStepReply, AgentAddress
import asyncio
import logging
from dataclasses import dataclass

"""
Agent responsible to synchronizing operation of all simulation agents (participants and central instance).
This is done (for now) via syncing messages.
For each simulation time step the Syncing agent sends each other agent a unique (by id) TimeStepMessage to trigger 
all its workload for this time step. 
When an agent has finished its computations it replies with a corresponding TimeStepReply.
Once all agents have replied for this time step the next time step can be triggered.
"""

TIME_INCREMENT = 15 * 60  # 15 minutes in seconds


@dataclass
class FutureAndID:
    fut: asyncio.Future
    c_id: int


class SyncingAgent(Agent):
    def __init__(self, container, agent_adresses):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)

        self.agent_adresses = agent_adresses
        self.conversation_id = 0
        self.step_futures = {}

    def handle_message(self, content, meta):
        # This method defines what the agent will do with incoming messages.
        sender_id = meta.get("sender_id", None)
        sender_addr = meta.get("sender_addr", None)
        sender = AgentAddress(sender_addr[0], sender_addr[1], sender_id)

        if isinstance(content, TimeStepReply):
            c_id = content.c_id
            expected_c_id = self.step_futures[sender].c_id
            sender_fut = self.step_futures[sender].fut

            if c_id == expected_c_id:
                sender_fut.set_result(True)
            else:
                logging.warn(
                    f"Got TimeStepReply with unexpected c_id: {c_id} --- expected: {expected_c_id}"
                )

    # times in seconds since epoch!
    async def run(self, start_time, end_time):
        # sanity check
        if start_time > end_time:
            raise ValueError("Start time after end time!")

        if start_time % TIME_INCREMENT != 0:
            raise ValueError("Start time is not on 15 minute slot!")

        if end_time % TIME_INCREMENT != 0:
            raise ValueError("End time is not on 15 minute slot!")

        # run each time step from start to end time in 15 minute increments
        current_time_step = start_time

        while current_time_step != end_time:
            await self.run_time_step(current_time_step)
            print("finished a time step!")
            current_time_step += TIME_INCREMENT

    async def run_time_step(self, time_step):
        futures = []

        for i, addr in enumerate(self.agent_adresses):
            # assuming we will always have fewer than 15 * 60 agents
            c_id = time_step + i
            fut = asyncio.Future()
            futures.append(fut)
            self.step_futures[addr] = FutureAndID(fut, c_id)
            self.send_sync_message(time_step, addr, c_id)

        # await all step futures
        await asyncio.gather(*futures)

    def send_sync_message(self, time_step, receiver, c_id):
        content = TimeStepMessage(time_step, c_id)
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}

        self.schedule_instant_acl_message(
            content,
            (receiver.host, receiver.port),
            receiver.agent_id,
            acl_metadata=acl_meta,
        )
