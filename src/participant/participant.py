from mango.agent.core import Agent
import pyomo.environ as pyo
import numpy as np
from messages.message_classes import TimeStepMessage, TimeStepReply, AgentAddress


import matplotlib.pyplot as plt

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
        # what should be done here?
        # recompute schedule every 24h
        # what else?
        pass

    def run(self):
        # to proactively do things
        pass

    def get_address(self):
        return AgentAddress(self.addr[0], self.addr[1], self.aid)


def calc_opt_day(
    forecasts: dict,
    pv_vals: dict,
    bss_vals: dict,
    cs_vals: dict,
    ev_vals: dict,
    elec_price=ELEC_PRICE,
    feedin_tariff=FEEDIN_TARIFF,
) -> dict:
    """Calculate the optimal schedule for the household based on profiles and parameters.

    Args:
        forecasts (dict): forecasts for devices profiles (baseload, pv, ev, hp)
        pv_vals (dict): Info about pv system - max power
        bss_vals (dict): Info about battery storage system - efficiency, max power and energy plus energy level
        cs_vals (dict): Info about charging station - efficiency and max power
        ev_vals (dict): Info about electric vehicle - max energy plus energy level
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
    model.x_p_peak = pyo.Var(range(1), domain=pyo.NonNegativeReals)
    model.x_p_valley = pyo.Var(range(1), domain=pyo.NonNegativeReals)
    model.x_pv_p = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, pv_vals["p_max"])
    )
    model.x_bss_p_charge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["p_max"])
    )
    model.x_bss_p_discharge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["p_max"])
    )
    model.x_bss_e = pyo.Var(
        DAY_STEPS_PLUS_1, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["e_max"])
    )
    model.x_cs_p_charge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, cs_vals["p_max"])
    )
    model.x_cs_p_discharge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, cs_vals["p_max"])
    )
    model.x_ev_e = pyo.Var(
        DAY_STEPS_PLUS_1, domain=pyo.NonNegativeReals, bounds=(0, ev_vals["e_max"])
    )
    model.x_ev_pen = pyo.Var(
        range(1), domain=pyo.NonNegativeReals, bounds=(0, ev_vals["e_max"])
    )

    # add objective function
    if isinstance(elec_price, float):
        elec_price = [elec_price for _ in DAY_STEPS]
    if isinstance(feedin_tariff, float):
        feedin_tariff = [feedin_tariff for _ in DAY_STEPS]
    ev_penalty = 10 * model.x_ev_pen[0]  # penalize non-filled ev battery
    ev_dir_cha = sum((t / 1e6 * model.x_cs_p_charge[t]) for t in range(len(DAY_STEPS)))
    c_bss_energy = (feedin_tariff[-1] + elec_price[-1]) / 2
    bss_incent = (
        -c_bss_energy * model.x_bss_e[DAY_STEPS_PLUS_1[-1]] / 1000
    )  # incentive bss energy at the end to not randomly feed into grid but store for next day
    c_peak = 0.01 / 1000  # peak price needs to be positive, height ist irrelevant
    c_valley = (
        min(c_bss_energy, np.mean(elec_price)) * GRANULARITY / 2 / 1000
    )  # to reduce valley instead of keeping energy and to not use grid energy to reduce valley
    model.OBJ = pyo.Objective(
        expr=sum(
            [
                elec_price[t] * model.x_grid_load[t] / 1000 * GRANULARITY
                - feedin_tariff[t] * model.x_grid_feedin[t] / 1000 * GRANULARITY
                for t in DAY_STEPS
            ]
        )
        + ev_penalty
        + bss_incent
        + ev_dir_cha
        + c_peak * model.x_p_peak[0]
        + c_valley * model.x_p_valley[0]
    )

    # add constraints: balance
    model.C_bal = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_bal.add(
            expr=forecasts["baseload"][t]
            + forecasts["hp_el_dem"][t]
            + model.x_cs_p_charge[t]
            + model.x_grid_feedin[t]
            + model.x_bss_p_charge[t]
            == model.x_pv_p[t] + model.x_grid_load[t] + model.x_bss_p_discharge[t]
        )

    # add constraints: peak and valley
    model.C_peak = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_peak.add(expr=model.x_p_peak[0] >= model.x_grid_load[t])
    model.C_val = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_val.add(expr=model.x_p_valley[0] >= model.x_grid_feedin[t])

    # add constraints: pv - max generation
    model.C_pv_power = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_pv_power.add(expr=model.x_pv_p[t] <= forecasts["pv_gen"][t])

    # add constraints: ev - time coupling & only charge while home & full at end of day
    model.C_ev_start = pyo.Constraint(expr=model.x_ev_e[0] == ev_vals["end"])
    model.C_ev_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_cs_p_charge[t] * cs_vals["eff"]
            - model.x_cs_p_discharge[t] / cs_vals["eff"]
        ) * GRANULARITY
        model.C_ev_coupl.add(
            expr=model.x_ev_e[t + 1]
            == model.x_ev_e[t] + e_adapt - forecasts["ev"]["consumption"][t]
        )
    model.C_ev_home = pyo.ConstraintList()
    for t in DAY_STEPS:
        if forecasts["ev"]["state"][t] != "home":
            model.C_ev_home.add(expr=model.x_cs_p_charge[t] == 0)
            model.C_ev_home.add(expr=model.x_cs_p_discharge[t] == 0)
    model.C_ev_end = pyo.Constraint(
        expr=model.x_ev_e[DAY_STEPS_PLUS_1[-1]] + model.x_ev_pen[0] == ev_vals["e_max"]
    )

    # add constraints: hp

    # add constraints: bss - time coupling
    model.C_bss_start = pyo.Constraint(expr=model.x_bss_e[0] == bss_vals["end"])
    model.C_bss_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_bss_p_charge[t] * bss_vals["eff"]
            - model.x_bss_p_discharge[t] / bss_vals["eff"]
        ) * GRANULARITY
        model.C_bss_coupl.add(expr=model.x_bss_e[t + 1] == model.x_bss_e[t] + e_adapt)

    try:  # try solving it with gurobi, otherwise use opensource solver HiGHS
        opt = pyo.SolverFactory("gurobi")
        opt.solve(model)
    except:
        opt = pyo.SolverFactory("appsi_highs")
        opt.solve(model)

    profiles = {}
    profiles["Baseload"] = forecasts["baseload"]
    profiles["PV"] = -forecasts["pv_gen"]
    profiles["EV"] = np.round(model.x_cs_p_charge[:](), 1) - np.round(
        model.x_cs_p_discharge[:](), 1
    )
    profiles["HP"] = forecasts["hp_el_dem"]
    profiles["BSS"] = np.round(model.x_bss_p_charge[:](), 1) - np.round(
        model.x_bss_p_discharge[:](), 1
    )
    profiles["BSS_e"] = np.round(model.x_bss_e[:](), 1)
    profiles["p_res"] = np.round(model.x_grid_load[:](), 1) - np.round(
        model.x_grid_feedin[:](), 1
    )
    if model.x_ev_pen[0]() > 0:
        print(f"Penalty variable for EV > 0: {np.round(model.x_ev_pen[0](), 1)}")

    return profiles
