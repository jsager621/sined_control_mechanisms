from datetime import datetime
import os
import pandas as pd
import numpy as np
import json
import random

from load_curves import LOAD_CURVES_DAY_REL

"""
Collection of utility functions for the simulation.
"""

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(THIS_DIR, "..", "data")
EV_DIR = os.path.join(DATA_DIR, "ev")
EV_FILE = os.path.join(DATA_DIR, "ev_kWh.csv")
HEATPUMP_FILE = os.path.join(DATA_DIR, "heatpump_el_kW.csv")
HOUSEHOLD_FILE = os.path.join(DATA_DIR, "household_load_kW.csv")
PV_FILE = os.path.join(DATA_DIR, "pv_10kWp_kW.csv")

CONFIG_DIR = os.path.join(THIS_DIR, "..", "config")
GRID_CONFIG = os.path.join(CONFIG_DIR, "grid.json")
PROSUMER_CONFIG = os.path.join(CONFIG_DIR, "prosumer.json")
SIMULATION_CONFIG = os.path.join(CONFIG_DIR, "simulation.json")

# Singleton class to ensure csv files are read only once per process
# regardless of which agents calls for data first.
class DataReader(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DataReader, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # read ev data in bulk from directory
        self.ev_data_sets = read_ev_data_sets()

        # read in the other data files
        raw_heatpump_data = pd.read_csv(HEATPUMP_FILE).to_numpy()
        raw_household_data = pd.read_csv(HOUSEHOLD_FILE).to_numpy()
        raw_pv_data = pd.read_csv(PV_FILE).to_numpy()

        # parse all timestamps to unix time
        

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

def read_ev_data_sets():
    ev_data_sets = []

    for file in os.listdir(EV_DIR):
        filename = os.fsdecode(file)
        if not filename.endswith(".csv"):
            continue

        filename = os.path.join(EV_DIR, filename)
        raw_ev_data = pd.read_csv(filename).to_numpy()
        # parse all timestamps to unix time
        raw_ev_data[:, 0] = [
            time_str_to_int(timestamp_str) for timestamp_str in raw_ev_data[:, 0]
        ]
        ev_data_sets.append(raw_ev_data)

    return ev_data_sets


def time_str_to_int(timestamp_str):
    # hack to enforce utc timestamp because it's annoying
    if not "+" in timestamp_str:
        timestamp_str += "+00:00"
    return int(datetime.fromisoformat(timestamp_str).timestamp())


def time_int_to_str(timestamp_int):
    return str(datetime.utcfromtimestamp(timestamp_int))

# read the ev corresponding to this agent number, determined by modulo on the 
# number of EV data sets we have
def read_ev_data(t_start, t_end, nr):
    reader = DataReader()
    n_data_sets = len(reader.ev_data_sets)
    np_data = reader.ev_data_sets[nr % n_data_sets]
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

def make_idealized_load_day(peak_min, peak_max, nr):
    n_load_days = len(LOAD_CURVES_DAY_REL)
    ideal_day_rel = LOAD_CURVES_DAY_REL[nr % n_load_days]

    peak = random.uniform(peak_min, peak_max)
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
