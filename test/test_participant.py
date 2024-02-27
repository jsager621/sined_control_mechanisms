import pytest
import numpy as np
from src.participant import calc_opt_day


def test_calc_opt_day():
    forecasts = {}
    forecasts["baseload"] = 500 * np.ones(96)
    forecasts["pv_gen"] = np.zeros(96)
    forecasts["pv_gen"][30:65] += 1000
    forecasts["pv_gen"][36:59] += 1000
    forecasts["pv_gen"][41:54] += 1200
    forecasts["pv_gen"][44:51] += 1000
    forecasts["pv_gen"][47:48] += 800
    forecasts["ev"] = {}
    forecasts["ev"]["state"] = ["home"] * 96
    forecasts["ev"]["consumption"] = np.zeros(96)
    forecasts["ev"]["state"][35] = "driving"
    for t in range(36, 70):
        forecasts["ev"]["state"][t] = "workplace"
    forecasts["ev"]["state"][70] = "driving"
    forecasts["ev"]["consumption"][35] = 5000
    forecasts["ev"]["consumption"][70] = 5000
    forecasts["hp_el_dem"] = np.zeros(96)
    forecasts["hp_el_dem"][10:20] = 2000
    forecasts["hp_el_dem"][40:45] = 2000
    forecasts["hp_el_dem"][80:90] = 2000
    pv_vals = {"p_max": 10000}
    bss_vals = {"end": 3000, "e_max": 5000, "p_max": 5000, "eff": 0.97}
    cs_vals = {"p_max": 11000, "eff": 0.95}
    ev_vals = {"end": 45000, "e_max": 45000}

    profiles = calc_opt_day(forecasts, pv_vals, bss_vals, cs_vals, ev_vals)

    print(profiles)
