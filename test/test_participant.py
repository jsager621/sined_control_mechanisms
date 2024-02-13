import pytest
import numpy as np
from src.participant import calc_opt_day


def test_calc_opt_day():
    baseload = 500 * np.ones(96)
    pv_gen = np.zeros(96)
    pv_gen[30:65] += 1000
    pv_gen[36:59] += 1000
    pv_gen[41:54] += 1200
    pv_gen[44:51] += 1000
    pv_gen[47:48] += 800
    ev_cha = np.zeros(96)
    ev_cha[70:75] = 11000
    hp_el_dem = np.zeros(96)
    hp_el_dem[10:20] = 2000
    hp_el_dem[40:45] = 2000
    hp_el_dem[80:90] = 2000
    bss_vals = {"end": 3000, "e_max": 5000, "p_max": 5000, "eff": 0.97}

    profiles = calc_opt_day(baseload, pv_gen, ev_cha, hp_el_dem, bss_vals)

    print(profiles)
