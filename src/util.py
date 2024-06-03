from datetime import datetime
import os
import pandas as pd
import numpy as np
import json
import random

"""
Collection of utility functions for the simulation.
"""

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(THIS_DIR, "..", "data")
EV_FILE = os.path.join(DATA_DIR, "ev_kWh.csv")
HEATPUMP_FILE = os.path.join(DATA_DIR, "heatpump_el_kW.csv")
HOUSEHOLD_FILE = os.path.join(DATA_DIR, "household_load_kW.csv")
PV_FILE = os.path.join(DATA_DIR, "pv_10kWp_kW.csv")

CONFIG_DIR = os.path.join(THIS_DIR, "..", "config")
GRID_CONFIG = os.path.join(CONFIG_DIR, "grid.json")
PROSUMER_CONFIG = os.path.join(CONFIG_DIR, "prosumer.json")
SIMULATION_CONFIG = os.path.join(CONFIG_DIR, "simulation.json")


ideal_day_rel = [ 
  0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.24, 0.22, 0.2, 0.2, 0.2, 0.2, 0.2, 0.22, 0.24, 0.25, 0.26, 0.28, 0.29, 0.30,
  0.31, 0.36, 0.41, 0.45, 0.50, 0.53, 0.58, 0.6, 0.6, 0.62, 0.65, 0.66, 0.68, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
  0.8, 0.8, 0.79, 0.72, 0.7, 0.68, 0.66, 0.62, 0.6, 0.58, 0.55, 0.54, 0.52, 0.5, 0.49, 0.51, 0.6, 0.68, 0.7, 0.75, 0.78, 0.8, 0.82, 0.84,
  0.86, 0.88, 0.9, 0.91, 0.92, 0.9, 0.9, 0.87, 0.84, 0.81, 0.78, 0.75, 0.72, 0.69, 0.66, 0.63, 0.60, 0.57, 0.54, 0.51, 0.48, 0.45, 0.42,  
  0.39, 0.36, 0.35
]


# Singleton class to ensure csv files are read only once per process
# regardless of which agents calls for data first.
class DataReader(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DataReader, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # read in the four data files
        raw_ev_data = pd.read_csv(EV_FILE).to_numpy()
        raw_heatpump_data = pd.read_csv(HEATPUMP_FILE).to_numpy()
        raw_household_data = pd.read_csv(HOUSEHOLD_FILE).to_numpy()
        raw_pv_data = pd.read_csv(PV_FILE).to_numpy()

        # parse all timestamps to unix time
        raw_ev_data[:, 0] = [
            time_str_to_int(timestamp_str) for timestamp_str in raw_ev_data[:, 0]
        ]
        self.ev_data = raw_ev_data

        raw_heatpump_data[:, 0] = [
            time_str_to_int(timestamp_str) for timestamp_str in raw_heatpump_data[:, 0]
        ]
        self.heatpump_data = raw_heatpump_data

        raw_household_data[:, 0] = [
            time_str_to_int(timestamp_str) for timestamp_str in raw_household_data[:, 0]
        ]
        self.household_data = raw_household_data

        raw_pv_data[:, 0] = [
            time_str_to_int(timestamp_str) for timestamp_str in raw_pv_data[:, 0]
        ]
        self.pv_data = raw_pv_data


def time_str_to_int(timestamp_str):
    # hack to enforce utc timestamp because it's annoying
    if not "+" in timestamp_str:
        timestamp_str += "+00:00"
    return int(datetime.fromisoformat(timestamp_str).timestamp())


def time_int_to_str(timestamp_int):
    return str(datetime.utcfromtimestamp(timestamp_int))


def read_ev_data(t_start, t_end):
    reader = DataReader()
    np_data = reader.ev_data
    mask = (np_data[:, 0] >= t_start) & (np_data[:, 0] < t_end)
    rows = np_data[mask]

    # return state and consumption as separate np arrays
    return (rows[:, 1].astype("S"), rows[:, 2].astype("f"))


def read_heatpump_data(t_start, t_end):
    reader = DataReader()
    np_data = reader.heatpump_data
    mask = (np_data[:, 0] >= t_start) & (np_data[:, 0] < t_end)
    rows = np_data[mask]

    # return P_TOT and Q_TOT as separate np arrays
    return (rows[:, 1].astype("f"), rows[:, 2].astype("f"))


def read_pv_data(t_start, t_end):
    reader = DataReader()
    np_data = reader.pv_data
    mask = (np_data[:, 0] >= t_start) & (np_data[:, 0] < t_end)
    rows = np_data[mask]

    # return P_IN_W
    return rows[:, 1].astype("f")


def read_load_data(t_start, t_end):
    reader = DataReader()
    np_data = reader.household_data
    mask = (np_data[:, 0] >= t_start) & (np_data[:, 0] < t_end)
    rows = np_data[mask]

    # return P_IN_W
    return rows[:, 1].astype("f")

def make_idealized_load_day(peak_min, peak_max):
    peak = random.random() * (peak_max - peak_min) + peak_min
    ideal = np.array(ideal_day_rel)
    return -1 * ideal * peak


def read_json(json_file):
    data = {}
    with open(json_file, "r") as f:
        data = json.load(f)

    return data


def read_prosumer_config():
    return read_json(PROSUMER_CONFIG)


def read_grid_config():
    return read_json(GRID_CONFIG)


def read_simulation_config():
    return read_json(SIMULATION_CONFIG)
