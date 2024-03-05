from mango.agent.core import Agent
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from messages.message_classes import TimeStepMessage, TimeStepReply, AgentAddress

# Pyomo released under 3-clause BSD license


class NetParticipant(Agent):
    def __init__(self, container):
        # We must pass a reference of the container to "mango.Agent":
        super().__init__(container)
        print(f"Hello world! My id is {self.aid}.")

        # initialize list to store result data
        self.result_timeseries_residual = []

        # TODO! Integrate the config data here in init

        # store length of one simulation step
        self.step_size_s = 15 * 60  # 15 min steps, 900 seconds

        # store tariff info
        self.tariff = {"electricity_price": 0.35, "feedin_tariff": 0.07}

        # store paths for the profiles (and corresponding columns)
        self.paths = {
            "load": "data/household_load.csv",
            "pv": "data/pv_10kw.csv",
            "ev": "data/ev_45kWh.csv",
            "hp": "data/heatpump.csv",
        }
        columns = {
            "load": "                P # [W]",
            "pv": "                P # [W]",
            "ev": {"state": "state", "consumption": "consumption"},
            "hp": "P_TOT",
        }

        # store forecast profiles
        self.forecast = {}
        self.forecast["load"] = (
            self.load_csv(path_to_csv=self.paths["load"], col_name=columns["load"])
            / 1000
        )
        self.forecast["pv"] = (
            self.load_csv(
                path_to_csv=self.paths["pv"], col_name=columns["pv"], n_header=0
            )
            / 1000
        )
        self.forecast["hp"] = (
            self.load_csv(path_to_csv=self.paths["hp"], col_name=columns["hp"]) / 1000
        )
        self.forecast["ev"] = {}
        self.forecast["ev"]["state"] = self.load_csv(
            path_to_csv=self.paths["ev"], col_name=columns["ev"]["state"], n_header=0
        )
        self.forecast["ev"]["consumption"] = (
            self.load_csv(
                path_to_csv=self.paths["ev"],
                col_name=columns["ev"]["consumption"],
                n_header=0,
            )
            / 1000
        )

        # store data of the devices
        self.dev = {}
        self.dev["pv"] = {"power_kWp": 10}
        self.dev["bss"] = {
            "capacity_kWh": 10,
            "power_kW": 10,
            "efficiency": 0.95,
            "e_kWh": 5,
        }
        self.dev["ev"] = {"capacity_kWh": 45, "e_kWh": 45}
        self.dev["cs"] = {"power_kW": 11, "efficiency": 0.95}

        # initialize residual schedule of the day with zeros for each value
        self.residual_schedule = np.zeros(int(3600 * 24 / self.step_size_s))
        # NOTE positive values are power demand, negative values are generation

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
        # retrieve forecasts for this day (with given timestamp and stepsize)
        # TODO!
        forecasts = {}
        forecasts["load"] = 0.5 * np.ones(96)
        forecasts["pv"] = np.zeros(96)
        forecasts["pv"][30:65] -= 1
        forecasts["pv"][36:59] -= 1
        forecasts["pv"][41:54] -= 1.2
        forecasts["pv"][44:51] -= 1
        forecasts["pv"][47:48] -= 0.8
        forecasts["ev"] = {}
        forecasts["ev"]["state"] = ["home"] * 96
        forecasts["ev"]["consumption"] = np.zeros(96)
        forecasts["ev"]["state"][35] = "driving"
        for t in range(36, 70):
            forecasts["ev"]["state"][t] = "workplace"
        forecasts["ev"]["state"][70] = "driving"
        forecasts["ev"]["consumption"][35] = 5
        forecasts["ev"]["consumption"][70] = 5
        forecasts["hp"] = np.zeros(96)
        forecasts["hp"][10:20] = 2
        forecasts["hp"][40:45] = 2
        forecasts["hp"][80:90] = 2

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

        print(f"Participant calculated for timestamp {timestamp}.")

    def run(self):
        # to proactively do things
        pass

    def get_address(self):
        return AgentAddress(self.addr[0], self.addr[1], self.aid)

    def load_csv(self, path_to_csv, col_name, n_header=1) -> np.ndarray:
        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=[0], header=n_header)
        profile_kW = df.loc[:, col_name].to_numpy()
        return profile_kW


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
