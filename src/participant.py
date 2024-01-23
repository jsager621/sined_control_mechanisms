# from mango import Agent
import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverFactory, SolverManagerFactory

from ortools.linear_solver import pywraplp

# Pyomo released under 3-clause BSD license

GRANULARITY = 0.25
DAY_STEPS = range(int(6 / GRANULARITY))
DAY_STEPS_PLUS_1 = range(int(24 / GRANULARITY + 1))
ELEC_PRICE = 0.35
FEEDIN_TARIFF = 0.07


class NetParticipant:  # Agent):
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


def calc_opt_day(load_profile, generation_profile, ev_profile, hp_profile, bss_vals):
    solver = pywraplp.Solver.CreateSolver("SAT")  # mip solver with SCIP backend
    inf = solver.infinity()

    # add optimization variables
    x_grid_load = {}
    x_grid_feedin = {}
    x_bss_p_charge = {}
    x_bss_p_discharge = {}
    x_bss_e = {}
    for t in DAY_STEPS:
        x_grid_load[t] = solver.NumVar(0, inf, "x_grid_load[%i]" % t)
        x_grid_feedin[t] = solver.NumVar(0, inf, "x_grid_feedin[%i]" % t)
        x_bss_p_charge[t] = solver.NumVar(
            0, bss_vals["p_max"], "x_bss_p_charge[%i]" % t
        )
        x_bss_p_discharge[t] = solver.NumVar(
            0, bss_vals["p_max"], "x_bss_p_discharge[%i]" % t
        )
        x_bss_e[t] = solver.NumVar(0, bss_vals["e_max"], "x_bss_e[%i]" % t)
    x_bss_e[t + 1] = solver.NumVar(0, bss_vals["e_max"], "x_bss_e[%i]" % t)
    print("Number of variables =", solver.NumVariables())

    # add objective function
    """obj_expr = [
        ELEC_PRICE * x_grid_load[t] - FEEDIN_TARIFF * x_grid_feedin[t]
        for t in DAY_STEPS
    ]
    solver.Minimize(solver.Sum(obj_expr))"""
    obj = solver.Objective()
    for t in DAY_STEPS:
        obj.SetCoefficient(x_grid_load[t], ELEC_PRICE / GRANULARITY / 1000)
        obj.SetCoefficient(x_grid_feedin[t], -FEEDIN_TARIFF / GRANULARITY / 1000)
        obj.SetMinimization()

    # add constraints: balance
    for t in DAY_STEPS:
        load_expr = [
            load_profile[t]
            + hp_profile[t]
            + ev_profile[t]
            + x_grid_feedin[t]
            + x_bss_p_charge[t]
        ]
        gen_expr = [generation_profile[t] + x_grid_load[t] + x_bss_p_discharge[t]]
        solver.Add(sum(load_expr) - sum(gen_expr) == 0)

    # add constraints: ev

    # add constraints: hp

    # add constraints: bss starting energy level
    solver.Add(x_bss_p_charge[0] == bss_vals["end"])
    # add constraints: bss time coupling
    for t in DAY_STEPS:
        adapt_e_expr = (
            x_bss_p_charge[t] * bss_vals["eff"] - x_bss_p_discharge[t] / bss_vals["eff"]
        ) * GRANULARITY
        solver.Add(x_bss_e[t + 1] == x_bss_e[t] + adapt_e_expr)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Objective value =", solver.Objective().Value())
        for t in DAY_STEPS:
            print(x_grid_load[t].name(), " = ", x_grid_load[t].solution_value())
            print(x_grid_feedin[t].name(), " = ", x_grid_feedin[t].solution_value())
            print(x_bss_p_charge[t].name(), " = ", x_bss_p_charge[t].solution_value())
            print(
                x_bss_p_discharge[t].name(),
                " = ",
                x_bss_p_discharge[t].solution_value(),
            )
            print(x_bss_e[t + 1].name(), " = ", x_bss_e[t + 1].solution_value())
        print()
        print(f"Problem solved in {solver.wall_time():d} milliseconds")
        print(f"Problem solved in {solver.iterations():d} iterations")
        print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")
    else:
        print("The problem does not have an optimal solution.")


# Test methods for Participant Class (shift later!)
if __name__ == "__main__":
    load_profile = 500 * np.ones(96)
    generation_profile = np.zeros(96)
    ev_profile = np.zeros(96)
    ev_profile[70:75] = 11000
    hp_profile = np.zeros(96)
    hp_profile[10:20] = 2000
    hp_profile[40:45] = 2000
    hp_profile[80:90] = 2000
    bss_vals = {"end": 3000, "e_max": 5000, "p_max": 5000, "eff": 0.97}

    calc_opt_day(load_profile, generation_profile, ev_profile, hp_profile, bss_vals)
