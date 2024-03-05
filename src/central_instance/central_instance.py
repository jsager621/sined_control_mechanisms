from mango import Agent
import random
import pandapower as pp
import pandapower.networks as ppnet
from messages.message_classes import TimeStepMessage, TimeStepReply, AgentAddress


class CentralInstance(Agent):
    grid = None  # pandapower net object
    grid_results = {}  # results of grid from powerflow calculation

    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)
        print(f"Hello world! My id is {self.aid}.")

        # initialize grid for simulation
        self.init_grid(grid_name="kerber_dorfnetz")

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
        self.num_participants = 57
        # TODO! Connection of participants and buses (if not everyone partic.)
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
        idx_loads = list(range(len(self.loadbus_names)))
        for i_part in range(self.num_participants):
            rnd_num = random.randint(0, len(idx_loads) - 1)
            self.load_participant_coord[str(i_part)] = idx_loads[rnd_num]
            idx_loads.pop(rnd_num)

        # store type of current control form
        self.control_type = "nothing"

    def handle_message(self, content, meta):
        # This method defines what the agent will do with incoming messages.
        sender_id = meta.get("sender_id", None)
        sender_addr = meta.get("sender_addr", None)
        sender = AgentAddress(sender_addr[0], sender_addr[1], sender_id)

        if isinstance(content, TimeStepMessage):
            c_id = content.c_id
            self.compute_time_step(content.time)

            # send reply
            content = TimeStepReply(c_id)
            acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}

            self.schedule_instant_acl_message(
                content,
                (sender.host, sender.port),
                sender.agent_id,
                acl_metadata=acl_meta,
            )

    def compute_time_step(self, timestamp):
        # collect agents residual schedules
        # TODO!
        # form: power_for_buses = {"loadbus_1_1": 2}
        power_for_buses = {
            self.grid.bus.loc[i_bus, "name"]: 1
            for i_bus in range(len(self.grid.bus))
            if "loadbus" in self.grid.bus.loc[i_bus, "name"]
        }

        # TODO maybe have to wait and ensure all agents have sent info
        # -> can be added with a corresponding await here as necessary

        # initialize schedule results list
        list_results_bus = {}
        list_results_line = {}
        for bus_name in self.result_timeseries_bus_vm_pu.keys():
            list_results_bus[bus_name] = []
        for line_name in self.result_timeseries_line_load.keys():
            list_results_line[line_name] = []

        # go by each time step (TODO! dynamically)
        for step in range(96):
            # set the inputs from the agents schedules
            self.set_inputs(data_for_buses=power_for_buses)

            # calculate the powerflow
            self.calc_grid_powerflow()

            # store the results of the powerflow
            self.store_grid_results()

            # set results into result saving lists
            for bus_name in self.grid_results_bus.keys():
                list_results_bus[bus_name].append(self.grid_results_bus[bus_name])
            for line_name in self.grid_results_line.keys():
                list_results_line[line_name].append(self.grid_results_line[line_name])

        # store list_results in result_timeseries JUST FOR LAST SCHEDULE CALCULATION
        # TODO!
        for bus_name in self.result_timeseries_bus_vm_pu.keys():
            self.result_timeseries_bus_vm_pu[bus_name].append(
                list_results_bus[bus_name]
            )
        for line_name in self.result_timeseries_line_load.keys():
            self.result_timeseries_line_load[line_name].append(
                list_results_line[line_name]
            )

        print(f"Central Instance calculated for timestamp {timestamp}.")

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
