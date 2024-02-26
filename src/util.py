from datetime import datetime

"""
Collection of utility functions for the simulation.
"""


def time_str_to_int(timestamp_str):
    return int(datetime.fromisoformat(timestamp_str).timestamp())


def read_charging_station(timestamp_int):
    pass


def read_heatpump(timestamp_int):
    pass


def read_household_load(timestamp_int):
    pass


def read_pv(timestamp_int):
    pass
