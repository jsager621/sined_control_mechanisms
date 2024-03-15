from mango.agent.core import Agent
import pyomo.environ as pyo
from pyomo.environ import NonNegativeReals as NNReals
from pyomo.environ import Var
import numpy as np
import asyncio
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
from util import (
    read_ev_data,
    read_pv_data,
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

        # initialize empty control signal from central instance
        self.control_signal: ControlMechanismMessage = ControlMechanismMessage()

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

        if isinstance(content, ControlMechanismMessage):
            logging.info(f"Got control message from agent: {sender}")

            # update information locally
            self.apply_control_message(content)

            # recalculate schedule based on new information
            self.compute_and_send_schedule(content.timestamp)

    def apply_control_message(self, content: ControlMechanismMessage):
        """Simply save a control signal that is sent."""
        self.control_signal = content

    def compute_and_send_schedule(self, timestamp):
        """Compute day-ahead schedule and send it via a message."""
        self.compute_day_ahead_schedule(timestamp)

        # inform central instance about residual schedule
        content = LocalResidualScheduleMessage(timestamp, self.residual_schedule)
        acl_meta = {"sender_id": self.aid, "sender_addr": self.addr}

        self.schedule_instant_acl_message(
            content,
            (self.central_agent.host, self.central_agent.port),
            self.central_agent.agent_id,
            acl_metadata=acl_meta,
        )

    def compute_time_step(self, timestamp):
        # retrieve forecasts for this day (with given timestamp and stepsize)
        # check if this is 00:00:00 on some day
        if timestamp % ONE_DAY_IN_SECONDS == 0:
            # remove any given control signal (first calculation for this day)
            self.control_signal = ControlMechanismMessage()

            self.compute_and_send_schedule(timestamp)
            logging.info(
                f"Participant {self.aid} calculated for timestamp {timestamp} --- {time_int_to_str(timestamp)}."
            )

    def read_forecast_data(self, timestamp) -> dict:
        t_start = timestamp
        t_end = timestamp + ONE_DAY_IN_SECONDS

        forecasts = {}
        forecasts["load"] = (
            read_load_data(t_start, t_end)
            * self.config["HOUSEHOLD"]["baseload_kWh_year"]
            / 4085
        )
        forecasts["pv"] = (
            read_pv_data(t_start, t_end) * self.dev["pv"]["power_kWp"] / 10
        )
        ev_state, ev_consumption = read_ev_data(t_start, t_end)
        ev_consumption = ev_consumption * self.dev["ev"]["capacity_kWh"] / 60
        forecasts["ev"] = {"state": ev_state, "consumption": ev_consumption}
        forecasts["hp"] = (
            read_heatpump_data(t_start, t_end)[0]
            * self.config["HOUSEHOLD"]["heatpump_kWh_el_year"]
            / 7042
        )

        return forecasts

    def compute_day_ahead_schedule(self, timestamp):
        forecasts = self.read_forecast_data(timestamp=timestamp)

        # compute schedule for upcoming day
        schedule = calc_opt_day(
            forecasts=forecasts,
            pv_vals=self.dev["pv"],
            bss_vals=self.dev["bss"],
            cs_vals=self.dev["cs"],
            ev_vals=self.dev["ev"],
            elec_price=self.tariff["electricity_price"],
            feedin_tariff=self.tariff["feedin_tariff"],
            step_size_s=self.step_size_s,
            control_sig=self.control_signal,
        )

        # retrieve residual schedule
        self.residual_schedule = schedule["p_res"]

        # retrieve energy level of BSS and EV at last time step (schedule["ev_e"][-1] &
        # schedule["bss_e"][-1]) JUST FOR LAST SCHEDULE CALCULATION
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
    control_sig: ControlMechanismMessage = ControlMechanismMessage(),
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
        control_sig (ControlMechanismMessage): message with control signal from central instance

    Returns:
        dict: schedules for the devices of the houeshold
    """
    # create parameters
    GRANULARITY = step_size_s / 3600
    DAY_STEPS = range(int(24 / GRANULARITY))
    DAY_STEPS_P1 = range(int(24 / GRANULARITY + 1))

    # initialize model
    model = pyo.ConcreteModel()

    # add optimization variables
    model.x_grid_load = Var(DAY_STEPS, domain=NNReals)
    model.x_grid_feedin = Var(DAY_STEPS, domain=NNReals)
    model.x_p_peak_load = Var(range(1), domain=NNReals)
    model.x_p_peak_gen = Var(range(1), domain=NNReals)
    model.x_pv_p = Var(DAY_STEPS, domain=NNReals, bounds=(0, pv_vals["power_kWp"]))
    model.x_bss_p_cha = Var(DAY_STEPS, domain=NNReals, bounds=(0, bss_vals["power_kW"]))
    model.x_bss_p_disch = Var(
        DAY_STEPS, domain=NNReals, bounds=(0, bss_vals["power_kW"])
    )
    model.x_bss_e = Var(
        DAY_STEPS_P1,
        domain=NNReals,
        bounds=(0, bss_vals["capacity_kWh"]),
    )
    model.x_cs_p_charge = Var(
        DAY_STEPS, domain=NNReals, bounds=(0, cs_vals["power_kW"])
    )
    model.x_cs_p_discharge = Var(
        DAY_STEPS, domain=NNReals, bounds=(0, cs_vals["power_discharge_kW"])
    )
    model.x_ev_e = Var(
        DAY_STEPS_P1,
        domain=NNReals,
        bounds=(0, ev_vals["capacity_kWh"]),
    )
    model.x_ev_pen = Var(range(1), domain=NNReals, bounds=(0, ev_vals["capacity_kWh"]))
    # variables for control signals regarding residual power limits
    model.control_pen_max = Var(DAY_STEPS, domain=NNReals)
    model.control_pen_min = Var(DAY_STEPS, domain=NNReals)

    # add objective function
    if isinstance(elec_price, float):
        elec_price = [elec_price for _ in DAY_STEPS]
    if isinstance(feedin_tariff, float):
        feedin_tariff = [feedin_tariff for _ in DAY_STEPS]
    ev_penalty = 10 * model.x_ev_pen[0]  # penalize non-filled ev battery
    ev_dir_cha = sum((t / 1e9 * model.x_cs_p_charge[t]) for t in range(len(DAY_STEPS)))
    c_bss_energy = (feedin_tariff[-1] + elec_price[-1]) / 2
    bss_incent = (
        -c_bss_energy * model.x_bss_e[DAY_STEPS_P1[-1]] * GRANULARITY
    )  # incentive bss energy at the end to not randomly feed into grid but store for next day
    # set peak prices in both directions for power peaks
    if control_sig is not None and control_sig.peak_price_dem is not None:
        c_peak_dem = control_sig.peak_price_dem
    else:
        c_peak_dem = 0.0
    if control_sig is not None and control_sig.peak_price_gen is not None:
        c_peak_gen = control_sig.peak_price_gen
    else:
        c_peak_gen = 0.0
    # penalize non-compliance with control signals
    obj_signals = (
        sum(  # simply add penalty costs to objective function if limits are surpassed
            100 * (model.control_pen_max[idx] + model.control_pen_min[idx])
            for idx in DAY_STEPS
        )
    )
    # adjust electricity price list with sent control signals
    if control_sig is not None and isinstance(control_sig.tariff_adj, np.ndarray):
        elec_price = list(elec_price + control_sig.tariff_adj)
    model.OBJ = pyo.Objective(
        expr=sum(
            [
                elec_price[t] * model.x_grid_load[t] * GRANULARITY
                - feedin_tariff[t] * model.x_grid_feedin[t] * GRANULARITY
                for t in DAY_STEPS
            ]
        )
        + obj_signals
        + ev_penalty
        + bss_incent
        + ev_dir_cha
        + c_peak_dem * model.x_p_peak_load[0]
        + c_peak_gen * model.x_p_peak_gen[0]
    )

    # add constraints: balance
    model.C_bal = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_bal.add(
            expr=forecasts["load"][t]
            + forecasts["hp"][t]
            + model.x_cs_p_charge[t]
            + model.x_grid_feedin[t]
            + model.x_bss_p_cha[t]
            == model.x_pv_p[t] + model.x_grid_load[t] + model.x_bss_p_disch[t]
        )

    # constraints for control signals: go by each separate signal and apply it via constraints
    if control_sig is not None:
        model.C_control = pyo.ConstraintList()
        for step in DAY_STEPS:
            # add constraint for max. and min. power demand at this step, add penalty variable
            if control_sig.p_max is not None and control_sig.p_max[step] != np.inf:
                model.C_control.add(
                    expr=model.x_grid_load[step] - model.x_grid_feedin[step]
                    <= control_sig.p_max[step] + model.control_pen_max[step]
                )
            if control_sig.p_min is not None and control_sig.p_min[step] != -np.inf:
                model.C_control.add(
                    expr=model.x_grid_load[step] - model.x_grid_feedin[step]
                    >= control_sig.p_min[step] - model.control_pen_min[step]
                )

    # add constraints: peak and valley
    model.C_peak = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_peak.add(expr=model.x_p_peak_load[0] >= model.x_grid_load[t])
    model.C_val = pyo.ConstraintList()
    for t in DAY_STEPS:
        model.C_val.add(expr=model.x_p_peak_gen[0] >= model.x_grid_feedin[t])

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
        if forecasts["ev"]["state"][t] != b"home":
            model.C_ev_home.add(expr=model.x_cs_p_charge[t] == 0)
            model.C_ev_home.add(expr=model.x_cs_p_discharge[t] == 0)
    model.C_ev_end = pyo.Constraint(
        expr=model.x_ev_e[DAY_STEPS_P1[-1]] + model.x_ev_pen[0]
        == ev_vals["capacity_kWh"]
    )

    # add constraints: bss - time coupling
    model.C_bss_start = pyo.Constraint(expr=model.x_bss_e[0] == bss_vals["e_kWh"])
    model.C_bss_coupl = pyo.ConstraintList()
    for t in DAY_STEPS:
        e_adapt = (
            model.x_bss_p_cha[t] * bss_vals["efficiency"]
            - model.x_bss_p_disch[t] / bss_vals["efficiency"]
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
        profiles["pv"] = -np.round(model.x_pv_p[:](), 4)
        profiles["ev"] = np.round(model.x_cs_p_charge[:](), 4) - np.round(
            model.x_cs_p_discharge[:](), 4
        )
        profiles["ev_e"] = np.round(model.x_ev_e[:](), 4)
        profiles["hp"] = forecasts["hp"]
        profiles["bss"] = np.round(model.x_bss_p_cha[:](), 4) - np.round(
            model.x_bss_p_disch[:](), 4
        )
        profiles["bss_e"] = np.round(model.x_bss_e[:](), 4)
        profiles["p_res"] = np.round(model.x_grid_load[:](), 4) - np.round(
            model.x_grid_feedin[:](), 4
        )
    else:
        raise ValueError(
            "Schedule Optimization unsuccessful: " + result.solver.termination_message
        )

    return profiles
