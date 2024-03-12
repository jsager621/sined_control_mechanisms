from mango.agent.core import Agent
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import asyncio
import logging

from messages.message_classes import (
    TimeStepMessage,
    TimeStepReply,
    AgentAddress,
    RegistrationMessage,
    RegistrationReply,
    LocalResidualScheduleMessage,
)
from util import (
    read_ev_data,
    read_pv_data,
    read_household_data,
    read_heatpump_data,
    read_load_data,
    read_prosumer_config,
    time_int_to_str,
)

ONE_DAY_IN_SECONDS = 24 * 60 * 60

# Pyomo released under 3-clause BSD license


class NetParticipant(Agent):
    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)

        self.central_agent = None
        self.registration_future = None

        # initialize list to store result data
        self.result_timeseries_residual = []

        # store length of one simulation step
        self.step_size_s = 15 * 60  # 15 min steps, 900 seconds

        # read config
        self.config = read_prosumer_config()
        self.tariff = self.config["TARIFF"]

        # store data of the devices
        self.dev = {}
        self.dev["pv"] = self.config["HOUSEHOLD"]["pv"]
        self.dev["bss"] = self.config["HOUSEHOLD"]["bss"]
        self.dev["ev"] = self.config["HOUSEHOLD"]["ev"]
        self.dev["cs"] = self.config["HOUSEHOLD"]["cs"]

        # initialize residual schedule of the day with zeros for each value
        self.residual_schedule = np.zeros(int(3600 * 24 / self.step_size_s))
        # NOTE positive values are power demand, negative values are generation

    async def register_to_central_agent(self, central_address):
        # send message
        content = RegistrationMessage()
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}
        self.registration_future = asyncio.Future()

        self.schedule_instant_acl_message(
            content,
            (central_address.host, central_address.port),
            central_address.agent_id,
            acl_metadata=acl_meta,
        )

        # await reply
        await self.registration_future
        logging.info("agent succesfully registered")

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

        if isinstance(content, RegistrationReply):
            if content.ack:
                self.central_agent = sender

            self.registration_future.set_result(True)

    def compute_time_step(self, timestamp):
        # retrieve forecasts for this day (with given timestamp and stepsize)
        # check if this is 00:00:00 on some day
        if timestamp % ONE_DAY_IN_SECONDS == 0:
            self.compute_day_ahead_schedule(timestamp)

        # inform central instance about residual schedule
        # TODO potentially wait out incoming messages that may change this for this time step?
        content = LocalResidualScheduleMessage(timestamp, self.residual_schedule)
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}

        self.schedule_instant_acl_message(
            content,
            (self.central_agent.host, self.central_agent.port),
            self.central_agent.agent_id,
            acl_metadata=acl_meta,
        )

        logging.info(
            f"Participant {self.aid} calculated for timestamp {timestamp} --- {time_int_to_str(timestamp)}."
        )

    def compute_day_ahead_schedule(self, timestamp):
        t_start = timestamp
        t_end = timestamp + ONE_DAY_IN_SECONDS

        forecasts = {}
        forecasts["ev"] = {}

        forecasts["load"] = read_load_data(t_start, t_end)
        forecasts["pv"] = read_pv_data(t_start, t_end)
        forecasts["ev"]["state"], forecasts["ev"]["consumption"] = read_ev_data(
            t_start, t_end
        )
        # only care about P_TOT here
        forecasts["hp"] = read_heatpump_data(t_start, t_end)[0]

        # compute schedule for upcoming day
        schedule = calc_opt_day(
            forecasts=forecasts,
            pv_vals=self.dev["pv"],
            bss_vals=self.dev["bss"],
            cs_vals=self.dev["cs"],
            ev_vals=self.dev["ev"],
            elec_price=self.tariff["electricity_price"],
            feedin_tariff=self.tariff["feedin_tariff"],
        )

        # retrieve residual schedule
        self.residual_schedule = schedule["p_res"]

        # retrieve energy level of BSS and EV at last time step (schedule["ev_e"] &
        # schedule["bss_e"]) JUST FOR LAST SCHEDULE CALCULATION
        # TODO!

        # update result_timeseries_residual at last calculation of this timestamp
        # TODO!
        self.result_timeseries_residual.append(self.residual_schedule)

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
    elec_price=0.35,
    feedin_tariff=0.07,
    step_size_s: int = 900,
) -> dict:
    """Calculate the optimal schedule for the household based on profiles and parameters.

    Args:
        forecasts (dict): forecasts for devices profiles (baseload, pv, ev, hp)
        pv_vals (dict): Info about pv system - max power
        bss_vals (dict): Info about battery storage system - efficiency, max power and energy plus energy level
        cs_vals (dict): Info about charging station - efficiency and max power
        ev_vals (dict): Info about electric vehicle - max energy plus energy level
        elec_price (list, float): Single value or list of values for grid electricity price.
            Defaults to 0.35.
        feedin_tariff (list, float): Single value or list of values for grid feedin price.
            Defaults to 0.07.
        step_size_s (int): Size of a simulation step in seconds. Defaults to 900 (=15 min).

    Returns:
        dict: schedules for the devices of the houeshold
    """
    # create parameters
    GRANULARITY = step_size_s / 3600
    DAY_STEPS = range(int(24 / GRANULARITY))
    DAY_STEPS_PLUS_1 = range(int(24 / GRANULARITY + 1))

    # initialize model
    model = pyo.ConcreteModel()

    # add optimization variables
    model.x_grid_load = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
    model.x_grid_feedin = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
    model.x_p_peak = pyo.Var(range(1), domain=pyo.NonNegativeReals)
    model.x_p_valley = pyo.Var(range(1), domain=pyo.NonNegativeReals)
    model.x_pv_p = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, pv_vals["power_kWp"])
    )
    model.x_bss_p_charge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["power_kW"])
    )
    model.x_bss_p_discharge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, bss_vals["power_kW"])
    )
    model.x_bss_e = pyo.Var(
        DAY_STEPS_PLUS_1,
        domain=pyo.NonNegativeReals,
        bounds=(0, bss_vals["capacity_kWh"]),
    )
    model.x_cs_p_charge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, cs_vals["power_kW"])
    )
    model.x_cs_p_discharge = pyo.Var(
        DAY_STEPS, domain=pyo.NonNegativeReals, bounds=(0, cs_vals["power_kW"])
    )
    model.x_ev_e = pyo.Var(
        DAY_STEPS_PLUS_1,
        domain=pyo.NonNegativeReals,
        bounds=(0, ev_vals["capacity_kWh"]),
    )
    model.x_ev_pen = pyo.Var(
        range(1), domain=pyo.NonNegativeReals, bounds=(0, ev_vals["capacity_kWh"])
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
        -c_bss_energy * model.x_bss_e[DAY_STEPS_PLUS_1[-1]]
    )  # incentive bss energy at the end to not randomly feed into grid but store for next day
    c_peak = 0.01  # peak price needs to be positive, height ist irrelevant
    c_valley = (
        min(c_bss_energy, np.mean(elec_price)) * GRANULARITY / 2
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
            expr=forecasts["load"][t]
            + forecasts["hp"][t]
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
        model.C_pv_power.add(expr=model.x_pv_p[t] <= -forecasts["pv"][t])

    # add constraints: ev - time coupling & only charge while home & full at end of day
    model.C_ev_start = pyo.Constraint(expr=model.x_ev_e[0] == ev_vals["e_kWh"])
    model.C_ev_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_cs_p_charge[t] * cs_vals["efficiency"]
            - model.x_cs_p_discharge[t] / cs_vals["efficiency"]
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
        expr=model.x_ev_e[DAY_STEPS_PLUS_1[-1]] + model.x_ev_pen[0]
        == ev_vals["capacity_kWh"]
    )

    # add constraints: hp

    # add constraints: bss - time coupling
    model.C_bss_start = pyo.Constraint(expr=model.x_bss_e[0] == bss_vals["e_kWh"])
    model.C_bss_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_bss_p_charge[t] * bss_vals["efficiency"]
            - model.x_bss_p_discharge[t] / bss_vals["efficiency"]
        ) * GRANULARITY
        model.C_bss_coupl.add(expr=model.x_bss_e[t + 1] == model.x_bss_e[t] + e_adapt)

    # solve the optimization problem
    solver = pyo.SolverFactory("gurobi")  # set solver as default to gurobi
    if not solver.available():  # check whether problem can be solved with gurobi
        solver = pyo.SolverFactory("appsi_highs")  # otherwise take OS solver
    # run the solver on the optimization problem
    result = solver.solve(model, load_solutions=False)

    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        # load the optimal results into the model
        model.solutions.load_from(result)
        profiles = {}
        profiles["load"] = forecasts["load"]
        profiles["pv"] = -np.round(model.x_pv_p[:](), 1)
        profiles["ev"] = np.round(model.x_cs_p_charge[:](), 1) - np.round(
            model.x_cs_p_discharge[:](), 1
        )
        profiles["ev_e"] = np.round(model.x_ev_e[:](), 1)
        profiles["hp"] = forecasts["hp"]
        profiles["bss"] = np.round(model.x_bss_p_charge[:](), 1) - np.round(
            model.x_bss_p_discharge[:](), 1
        )
        profiles["bss_e"] = np.round(model.x_bss_e[:](), 1)
        profiles["p_res"] = np.round(model.x_grid_load[:](), 1) - np.round(
            model.x_grid_feedin[:](), 1
        )
        if model.x_ev_pen[0]() > 0:
            print(f"Penalty variable for EV > 0: {np.round(model.x_ev_pen[0](), 1)}")
    else:
        raise ValueError(
            "Schedule Optimization unsuccessful: " + result.solver.termination_message
        )

    return profiles
