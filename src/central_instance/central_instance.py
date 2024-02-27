from mango import Agent
import pandapower as pp
import pandapower.networks as ppnet


class CentralInstance(Agent):
    grid = None  # pandapower net object
    grid_results = {}  # results of grid from powerflow calculation

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

    def init_grid(self, grid_name: str):
        if grid_name == "kerber_dorfnetz":
            self.grid = ppnet.create_kerber_dorfnetz()
        elif grid_name == "kerber_landnetz":
            self.grid = ppnet.create_kerber_landnetz_kabel_1()
        else:
            print(f"Grid {self.aid} could not be created.")
            self.grid = None

    def set_inputs(self, data_for_buses):
        """Set active/reactive power values for the grid load buses."""
        for name_bus, power_values in data_for_buses.items():
            element = getattr(self.grid, name_bus)

            element.at[name_bus, "p_mw"] = power_values[0]  # active power value
            element.at[name_bus, "q_mvar"] = power_values[1]  # reactive power value

    def calc_grid_powerflow(self):
        pp.runpp(
            self.grid,
            numba=False,
            calculate_voltage_angles=False,
        )

        self.grid_results = {}
        if not self.grid.res_bus.empty:  # powerflow converged
            for i_bus in range(len(self.grid.res_bus)):
                self.grid_results[i_bus] = self.grid.res_bus.loc[i_bus]
            for i_line in range(len(self.grid.res_line)):
                self.grid_results[i_line] = self.grid.res_line.loc[i_line]

    def run(self):
        # to proactively do things
        pass
