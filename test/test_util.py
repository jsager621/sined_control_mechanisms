from src.util import (
    read_ev_data,
    read_household_data,
    read_heatpump_data,
    read_pv_data,
    read_prosumer_config,
    read_grid_config,
    read_simulation_config,
)


def test_json_readers():
    read_prosumer_config()
    read_grid_config()
    read_simulation_config()


def test_data_readers():
    t_start = 1577844900
    t_end = 1577844900 + 4 * 15 * 60

    read_ev_data(t_start, t_end)
    read_heatpump_data(t_start, t_end)
    read_household_data(t_start, t_end)
    read_pv_data(t_start, t_end)
