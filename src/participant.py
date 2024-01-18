from mango import Agent
import pyomo.environ as pyo

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

    def calc_opt_day(
        self, load_profile, generation_profile, ev_profile, hp_profile, bss_vals
    ):
        model = pyo.ConcreteModel()

        # add optimization variables
        model.x_grid_load = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
        model.x_grid_feedin = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
        model.x_bss_p_charge = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
        model.x_bss_p_discharge = pyo.Var(DAY_STEPS, domain=pyo.NonNegativeReals)
        model.x_bss_e = pyo.Var(DAY_STEPS_PLUS_1, domain=pyo.NonNegativeReals)

        # add objective function
        model.OBJ = pyo.Objective(
            expr=ELEC_PRICE * model.x_grid_load[:]
            - FEEDIN_TARIFF * model.x_grid_feedin[:]
        )

        # add constraints: balance
        model.Constraint1 = pyo.Constraint(
            expr=load_profile
            + hp_profile
            + ev_profile
            + model.x_grid_feedin[:]
            + model.x_bss_p_charge[:]
            == generation_profile + model.x_grid_load[:] + model.x_bss_p_discharge[:]
        )

        # add constraints: ev

        # add constraints: hp

        # add constraints: bss
        opt_prob.addConstr(
            (opt_vars["bat_energy_kWh"][0] == 0)
        )  # BSS empty at the beginning
        for t in DAY_STEPS:
            opt_prob.addConstrs(
                (
                    opt_vars["bat_energy_kWh"][t + 1]
                    == opt_vars["bat_energy_kWh"][t]
                    + (
                        opt_vars["bat_charge_kW"][t] * bss_vals["eff"]
                        - opt_vars["bat_discharge_kW"][t] / bss_vals["eff"]
                    )
                    * GRANULARITY
                    for t in DAY_STEPS
                ),
            )
        opt_prob.addConstrs(
            (
                opt_vars["bat_energy_kWh"][t] <= bss_vals["cap_kWh"]
                for t in DAY_STEPS_PLUS_1
            ),
        )
        opt_prob.addConstrs(
            (opt_vars["bat_charge_kW"][t] <= bss_vals["power_kW"] for t in DAY_STEPS),
        )
        opt_prob.addConstrs(
            (
                opt_vars["bat_discharge_kW"][t] <= bss_vals["power_kW"]
                for t in DAY_STEPS
            ),
        )

    def run(self):
        # to proactively do things
        pass
