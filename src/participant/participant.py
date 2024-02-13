from mango.agent.core import Agent
import pyomo.environ as pyo
import numpy as np

# Pyomo released under 3-clause BSD license

GRANULARITY = 0.25
DAY_STEPS = range(int(24 / GRANULARITY))
DAY_STEPS_PLUS_1 = range(int(24 / GRANULARITY + 1))
ELEC_PRICE = 0.35
FEEDIN_TARIFF = 0.07


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


def calc_opt_day(
    baseload: list,
    pv_gen: list,
    ev_cha: list,
    hp_el_dem: list,
    bss_vals: dict,
    elec_price=ELEC_PRICE,
    feedin_tariff=FEEDIN_TARIFF,
) -> dict:
    """Calculate the optimal schedule for the household based on profiles and parameters.

    Args:
        baseload (list): baseload power demand profile
        pv_gen (list): pv power generation profile
        ev_cha (list): ev charging power profile
        hp_el_dem (list): heatpump power demand profiles
        bss_vals (dict): Info about bss - max power and energy plus energy level
        elec_price (list, float): Single value or list of values for grid electricity price.
            Defaults to ELEC_PRICE.
        feedin_tariff (list, float): Single value or list of values for grid feedin price.
            Defaults to FEEDIN_TARIFF.

    Returns:
        dict: schedules for the devices of the houeshold
    """
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
    if isinstance(elec_price, float):
        elec_price = [elec_price for _ in DAY_STEPS]
    if isinstance(feedin_tariff, float):
        feedin_tariff = [feedin_tariff for _ in DAY_STEPS]
    model.OBJ = pyo.Objective(
        expr=sum(
            [
                elec_price[t] * model.x_grid_load[t] / 1000 / GRANULARITY
                - feedin_tariff[t] * model.x_grid_feedin[t] / 1000 / GRANULARITY
                for t in DAY_STEPS
            ]
        )
    )

    # add constraints: balance
    model.C_bal = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_bal.add(
            expr=baseload[t]
            + hp_el_dem[t]
            + ev_cha[t]
            + model.x_grid_feedin[t]
            + model.x_bss_p_charge[t]
            == pv_gen[t] + model.x_grid_load[t] + model.x_bss_p_discharge[t]
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
        opt = pyo.SolverFactory("gurobi")
        log = opt.solve(model)
    except:
        opt = pyo.SolverFactory("appsi_highs")
        log = opt.solve(model)

    profiles = {}
    profiles["Baseload"] = baseload
    profiles["PV"] = -pv_gen
    profiles["EV"] = ev_cha
    profiles["HP"] = hp_el_dem
    profiles["BSS"] = np.round(model.x_bss_p_charge[:](), 1) - np.round(
        model.x_bss_p_discharge[:](), 1
    )

    return profiles
