from datetime import datetime
import os
import pandas as pd
import logging

"""
Collection of utility functions for the simulation.
"""

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")

EV_FILE = os.path.join(DATA_DIR, "ev_45kWh.csv")
HEATPUMP_FILE = os.path.join(DATA_DIR, "heatpump.csv")
HOUSEHOLD_FILE = os.path.join(DATA_DIR, "household_load.csv")
PV_FILE = os.path.join(DATA_DIR, "pv_10kw.csv")


# Singleton class to ensure csv files are read only once per process
# regardless of which agents calls for data first.
class DataReader(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DataReader, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # read in the four data files
        self.ev_data = pd.read_csv(EV_FILE)
        self.heatpump_data = pd.read_csv(HEATPUMP_FILE)
        self.household_data = pd.read_csv(HOUSEHOLD_FILE)
        self.pv_data = pd.read_csv(PV_FILE)


def time_str_to_int(timestamp_str):
    return int(datetime.fromisoformat(timestamp_str).timestamp())


def time_int_to_str(timestamp_int):
    return str(datetime.utcfromtimestamp(timestamp_int))


def read_ev_data(timestamp_int):
    reader = DataReader()
    str_timestamp = time_int_to_str(timestamp_int)
    row_data = reader.ev_data.loc[reader.ev_data["date"] == str_timestamp]

    if row_data.empty:
        raise ValueError(
            f"Tried to read ev data with unknown timestamp: {str_timestamp}"
        )

    return (
        row_data["date"].item(),
        row_data["state"].item(),
        row_data["consumption"].item(),
    )


def read_heatpump_data(timestamp_int):
    reader = DataReader()
    str_timestamp = time_int_to_str(timestamp_int)
    row_data = reader.heatpump_data.loc[reader.heatpump_data["date"] == str_timestamp]

    if row_data.empty:
        raise ValueError(
            f"Tried to read heatpump data with unknown timestamp: {str_timestamp}"
        )

    return (
        row_data["date"].item(),
        row_data["P_TOT"].item(),
        row_data["Q_TOT"].item(),
    )


def read_household_data(timestamp_int):
    reader = DataReader()
    str_timestamp = time_int_to_str(timestamp_int)
    row_data = reader.household_data.loc[reader.household_data["date"] == str_timestamp]

    if row_data.empty:
        raise ValueError(
            f"Tried to read household data with unknown timestamp: {str_timestamp}"
        )

    return (
        row_data["date"].item(),
        row_data["P_IN_W"].item(),
    )


def read_pv_data(timestamp_int):
    reader = DataReader()
    str_timestamp = time_int_to_str(timestamp_int)
    row_data = reader.pv_data.loc[reader.pv_data["date"] == str_timestamp]

    if row_data.empty:
        raise ValueError(
            f"Tried to read pv data with unknown timestamp: {str_timestamp}"
        )

    return (
        row_data["date"].item(),
        row_data["P_IN_W"].item(),
    )


if __name__ == "__main__":
    print(read_ev_data(1577844900))
    print(read_heatpump_data(1577844900))
    print(read_household_data(1577844900))
    print(read_pv_data(1577844900))
