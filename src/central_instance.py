from mango import Agent
import pandapower

class CentralInstance(Agent):
    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)
        print(f"Hello world! My id is {self.aid}.")

    def handle_message(self, content, meta):
        # This method defines what the agent will do with incoming messages.
        print(f"Received a message with the following content: {content}")

    def _send_price_message(self, target, content):
        pass

    def _send_schedule_message(self, target, content):
        pass

    def run(self):
        # to proactively do things
        pass