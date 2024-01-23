# from mango import Agent
import pyomo.environ as pyo
import numpy as np
from pyomo.contrib import appsi

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
    model = pyo.ConcreteModel()

    # add optimization variables
    model.x_grid_load = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
    model.x_grid_feedin = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
    model.x_bss_p_charge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["p_max"])
    )
    model.x_bss_p_discharge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["p_max"])
    )
    model.x_bss_e = pyo.Var(
        DAY_STEPS_PLUS_1, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["e_max"])
    )

    # add objective function
    model.OBJ = pyo.Objective(
        expr=sum(
            [
                ELEC_PRICE * model.x_grid_load[t] / 1000 / GRANULARITY
                - FEEDIN_TARIFF * model.x_grid_feedin[t] / 1000 / GRANULARITY
                for t in DAY_STEPS
            ]
        )
    )

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

    # add constraints: bss starting energy level - empty at beginning
    model.C_bss_start = pyo.Constraint(expr=model.x_bss_e[0] == bss_vals["end"])
    # add constraints: bss time coupling
    model.C_bss_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_bss_p_charge[t] * bss_vals["eff"]
            - model.x_bss_p_discharge[t] / bss_vals["eff"]
        ) * GRANULARITY
        model.C_bss_coupl.add(expr=model.x_bss_e[t + 1] == model.x_bss_e[t] + e_adapt)

    try:  # try solving it with gurobi, otherwise use opensource solver HiGHS
        opt = appsi.solvers.Gurobi()
        opt = pyo.SolverFactory("gurobi")
        log = opt.solve(model)
    except:
        opt = pyo.SolverFactory("appsi_highs")
        log = opt.solve(model)

    opt = pyo.SolverFactory("appsi_highs")  # try also 'cbc', 'glpk', 'gurobi'
    # log = opt.solve(model, tee=True)
    # log.write()
    # assert log.solver.status == pyo.SolverStatus.ok
    print("Objective value =", model.OBJ())
    print("grid_load: ", np.round(model.x_grid_load[:](), 1))
    print("grid_feedin: ", np.round(model.x_grid_feedin[:](), 1))
    print("bss_p_charge: ", np.round(model.x_bss_p_charge[:](), 1))
    print("bss_p_discharge: ", np.round(model.x_bss_p_discharge[:](), 1))
    print("bss_e: ", np.round(model.x_bss_e[:](), 1))


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
