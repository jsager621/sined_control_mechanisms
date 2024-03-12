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
    read_ev_data(1577844900)
    read_heatpump_data(1577844900)
    read_household_data(1577844900)
    read_pv_data(1577844900)
