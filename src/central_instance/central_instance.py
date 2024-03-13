from mango import Agent
import asyncio
import pandapower as pp
import pandapower.networks as ppnet
import logging
from messages.message_classes import (
    TimeStepMessage,
    TimeStepReply,
    AgentAddress,
    RegistrationMessage,
    RegistrationReply,
    LocalResidualScheduleMessage,
    ControlMechanismMessage,
)
from util import time_int_to_str, read_grid_config

# Pandapower Copyright (c) 2018 by University of Kassel and Fraunhofer IEE

ONE_DAY_IN_SECONDS = 24 * 60 * 60


class CentralInstance(Agent):
    grid = None  # pandapower net object
    grid_results = {}  # results of grid from powerflow calculation

    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)
        self.grid_config = read_grid_config()

        # initialize grid for simulation
        self.init_grid(grid_name=self.grid_config["GRID"])
        self.load_bus_names = [x for x in self.grid.bus["name"] if x.startswith("load")]

        # initialize list to store result data for buses and lines
        self.result_timeseries_bus_vm_pu = {}
        self.result_timeseries_line_load = {}
        for i_bus in range(len(self.grid.bus)):
            self.result_timeseries_bus_vm_pu[self.grid.bus.loc[i_bus, "name"]] = []
        for i_line in range(len(self.grid.line)):
            self.result_timeseries_line_load[self.grid.line.loc[i_line, "name"]] = []
        for i_trafo in range(len(self.grid.trafo)):
            self.result_timeseries_line_load[self.grid.trafo.loc[i_trafo, "name"]] = []

        # store participants and their connection to the buses
        self.num_households = self.grid_config["NUM_HOUSEHOLDS"]
        self.num_participants = self.grid_config["NUM_PARTICIPANTS"]
        self.current_participants = 0

        if self.num_households > len(self.load_bus_names):
            raise ValueError(
                "Trying to create grid with more households than load buses!"
            )

        if self.num_participants > self.num_households:
            raise ValueError(
                "Trying to create grid with more participants than households!"
            )

        self.loadbus_names = [
            self.grid.bus.loc[i_bus, "name"]
            for i_bus in range(len(self.grid.bus))
            if "loadbus" in self.grid.bus.loc[i_bus, "name"]
        ]

        if len(self.loadbus_names) < self.num_participants:
            raise ValueError(
                f"{self.num_participants} for {len(self.loadbus_names)} load buses does not work!"
            )

        self.load_participant_coord = {}
        self.received_schedules = {}
        self.time_step_done = False

        # store type of current control form
        self.control_type = self.grid_config["CONTROL_TYPE"]

    def handle_message(self, content, meta):
        # This method defines what the agent will do with incoming messages.
        sender_id = meta.get("sender_id", None)
        sender_addr = meta.get("sender_addr", None)
        sender = AgentAddress(sender_addr[0], sender_addr[1], sender_id)
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}

        if isinstance(content, TimeStepMessage):
            self.schedule_instant_task(
                self.compute_time_step(content.time, sender, content.c_id)
            )

        # no need to sync this because we run it synchronized from the simulation setup file
        # if this was truly asynchronous we would have to safeguard it
        if isinstance(content, RegistrationMessage):
            if self.current_participants < self.num_participants:
                self.add_participant(sender)
                content = RegistrationReply(True)
            else:
                content = RegistrationReply(False)

            self.schedule_instant_acl_message(
                content,
                (sender.host, sender.port),
                sender.agent_id,
                acl_metadata=acl_meta,
            )

        if isinstance(content, LocalResidualScheduleMessage):
            logging.info(f"Got residual schedule message from agent: {sender}")

            if content.timestamp not in self.received_schedules.keys():
                self.received_schedules[content.timestamp] = {}

            self.received_schedules[content.timestamp][
                sender
            ] = content.residual_schedule

    def add_participant(self, participant_address):
        if self.current_participants < self.num_participants:
            self.current_participants += 1
            self.load_participant_coord[participant_address] = self.load_bus_names[
                self.current_participants
            ]
        else:
            logging.warn(
                "Trying to register too many participants. Excess participants will be ignored by the central instance."
            )

    async def get_participant_schedules(self, timestamp):
        # TODO make this nicer with futures?
        while timestamp not in self.received_schedules.keys():
            await asyncio.sleep(0.01)

        while len(self.received_schedules[timestamp].keys()) < self.num_participants:
            await asyncio.sleep(0.01)

    def check_schedule_ok(self, aggregate_schedule):
        return True

    async def calculate_aggregate_schedule(self, timestamp):
        # NOTE:
        # Assumes all relevant participant schedules are now present in:
        # self.received_schedules[timestamp]
        #
        #
        # self.load_participant_coord[participant_address] - maps participant address to bus number
        # self.received_schedules[timestamp][participan_address] - contains the corresponding schedule
        #
        # for simplicity, parse these two things into one dict here:
        bus_to_schedule = {}
        for addr, bus_id in self.load_participant_coord.items():
            bus_to_schedule[bus_id] = self.received_schedules[timestamp][addr]

        # form: power_for_buses = {"loadbus_1_1": 2}

        aggregate_schedule = None
        self.time_step_done = self.check_schedule_ok(aggregate_schedule)

    def clear_local_schedules(self, timestamp):
        if timestamp in self.received_schedules.keys():
            del self.received_schedules[timestamp]

    async def apply_control_mechanisms(self, timestamp):
        pass

    def dummy_mechanism(self, timestamp):
        # just a "do nothing" control message
        # can be removed later, just here to check the
        # side of the logic responsible for sending these out and
        # incorporating them!
        pass

    def send_time_step_done_to_syncing_agent(self, sender, c_id):
        # send reply
        content = TimeStepReply(c_id)
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}
        self.schedule_instant_acl_message(
            content,
            (sender.host, sender.port),
            sender.agent_id,
            acl_metadata=acl_meta,
        )

    async def compute_time_step(self, timestamp, sender, c_id):
        # collect agents residual schedules
        if timestamp % ONE_DAY_IN_SECONDS == 0:
            self.time_step_done = False

            # always happens at least once
            await self.get_participant_schedules(timestamp)
            await self.calculate_aggregate_schedule(timestamp)

            # gets instantly skipped if schedules are already ok
            # flag gets set by calculate_aggregate_schedule when the schedule
            # fulfilly all requirements
            while not self.time_step_done:
                # NOTE: runs the risk of infinitely looping if control mechanisms
                # do not converge on a viable solution!
                # TODO: maybe consider max number of retries or ensure there is always
                # a strong enough final mechanism to ensure compliance
                self.clear_local_schedules(timestamp)
                await self.apply_control_mechanisms(timestamp)
                await self.get_participant_schedules(timestamp)
                await self.calculate_aggregate_schedule(timestamp)

            self.send_time_step_done_to_syncing_agent(sender, c_id)

            logging.info(
                f"Central Instance calculated for timestamp {timestamp} --- {time_int_to_str(timestamp)}."
            )

    def init_grid(self, grid_name: str):
        if grid_name == "kerber_dorfnetz":
            self.grid = ppnet.create_kerber_dorfnetz()
        elif grid_name == "kerber_landnetz":
            self.grid = ppnet.create_kerber_landnetz_kabel_1()
        else:
            print(f"Grid {self.aid} could not be created.")
            self.grid = None

    def set_inputs(self, data_for_buses):
        """Set active/reactive power values for the grid loads."""
        for name_bus, power_value in data_for_buses.items():
            # get the index of the bus
            idx_b = self.grid.bus.index[self.grid.bus["name"] == name_bus].to_list()[0]
            # get the index of the load and the load element
            idx_l = self.grid.load.index[self.grid.load["bus"] == idx_b].to_list()[0]
            element = self.grid.load.loc[idx_l]

            # set active power (and remove reactive power value)
            element.at["p_mw"] = power_value / 1000  # active power in MW
            element.at["q_mvar"] = 0  # reactive power value

    def calc_grid_powerflow(self):
        pp.runpp(
            self.grid,
            numba=False,
            calculate_voltage_angles=False,
        )

    def store_grid_results(self):
        self.grid_results_bus = {}
        self.grid_results_line = {}
        if not self.grid.res_bus.empty:  # powerflow converged
            for i_bus in range(len(self.grid.bus)):
                self.grid_results_bus[self.grid.bus.loc[i_bus, "name"]] = (
                    self.grid.res_bus.loc[i_bus, "vm_pu"]
                )
            for i_line in range(len(self.grid.line)):
                self.grid_results_line[self.grid.line.loc[i_line, "name"]] = (
                    self.grid.res_line.loc[i_line, "loading_percent"]
                )
            for i_trafo in range(len(self.grid.trafo)):
                self.grid_results_line[self.grid.trafo.loc[i_trafo, "name"]] = (
                    self.grid.res_trafo.loc[i_trafo, "loading_percent"]
                )

    def setup_control_signal(self):
        # TODO! With regard to type of control
        self.control_signal = {}

    def run(self):
        # to proactively do things
        pass

    def get_address(self):
        return AgentAddress(self.addr[0], self.addr[1], self.aid)
