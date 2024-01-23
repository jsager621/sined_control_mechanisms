# from mango import Agent
import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverFactory, SolverManagerFactory

from ortools.linear_solver import pywraplp

# Pyomo released under 3-clause BSD license

GRANULARITY = 0.25
DAY_STEPS = range(int(24 / GRANULARITY))
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
    model = pyo.ConcreteModel()
    solver = pywraplp.Solver.CreateSolver("SAT")  # mip solver with SCIP backend
    inf = solver.infinity()

    # add optimization variables
    x_grid_load = {}
    x_grid_feedin = {}
    x_bss_p_charge = {}
    x_bss_p_discharge = {}
    x_bss_e = {}
    for step in DAY_STEPS:
        x_grid_load[step] = solver.NumVar(0, inf, "grid_load[%i]" % step)
        x_grid_feedin[step] = solver.NumVar(0, inf, "x_grid_feedin[%i]" % step)
        x_bss_p_charge[step] = solver.NumVar(0, inf, "x_bss_p_charge[%i]" % step)
        x_bss_p_discharge[step] = solver.NumVar(0, inf, "x_bss_p_discharge[%i]" % step)
        x_bss_e[step] = solver.NumVar(0, bss_vals["e_max"], "x_bss_e[%i]" % step)
    x_bss_e[step + 1] = solver.NumVar(0, bss_vals["e_max"], "x_bss_e[%i]" % step)
    print("Number of variables =", solver.NumVariables())

    # add objective function
    """obj_expr = [
        ELEC_PRICE * x_grid_load[step] - FEEDIN_TARIFF * x_grid_feedin[step]
        for step in DAY_STEPS
    ]
    solver.Minimize(solver.Sum(obj_expr))"""
    obj = solver.Objective()
    for step in DAY_STEPS:
        obj.SetCoefficient(x_grid_load[step], ELEC_PRICE)
        obj.SetCoefficient(x_grid_feedin[step], FEEDIN_TARIFF)
        obj.SetMinimization()

    # add constraints: balance
    model.C_bal = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_bal.add(
            expr=load_profile[t]
            + hp_profile[t]
            + ev_profile[t]
            + model.x_grid_feedin[t]
            + model.x_bss_p_charge[t]
            == generation_profile[t] + model.x_grid_load[t] + model.x_bss_p_discharge[t]
        )

    # add constraints: ev

    # add constraints: hp

    # add constraints: bss starting energy level
    solver.Add(x_bss_p_charge[0] == bss_vals["end"])
    model.C_bss_start = pyo.Constraint(expr=model.x_bss_p_charge[0] == bss_vals["end"])
    # add constraints: bss time coupling
    model.C_bss_time_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_bss_time_coupl.add(
            expr=model.x_bss_e[t + 1]
            == model.x_bss_e[t]
            + (
                model.x_bss_p_charge[t] * bss_vals["eff"]
                - model.x_bss_p_discharge[t] / bss_vals["eff"]
            )
            * GRANULARITY,
        )
    # add constraints: bss limit maximum power
    model.C_bss_pcha_lim = pyo.Constraint(
        expr=model.x_bss_p_charge <= bss_vals["p_max"]
    )
    model.C_bss_pdischa_lim = pyo.Constraint(
        expr=model.x_bss_p_discharge <= bss_vals["p_max"]
    )

    opt = SolverFactory("ipopt")
    opt.set_options("max_iter=2")
    solver_manager = SolverManagerFactory("serial")
    results = solver_manager.solve(model, opt=opt, tee=True, timelimit=None)
    model.load(results)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()


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
