from mango import Agent
import pandapower

class NetParticipant(Agent):
    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)
        print(f"Hello world! My id is {self.aid}.")

        # TODO: units?
        self._feed_in = 0
        self._load = 0

        # TODO presumably we can directly use some pandapower components here to abstract the actual assets for us?

    def handle_message(self, content, meta):
        # This method defines what the agent will do with incoming messages.
        print(f"Received a message with the following content: {content}")

    def _handle_price_message(self, content, meta):
        pass

    def _handle_schedule_message(self, content, meta):
        pass

    def run(self):
        # to proactively do things
        pass