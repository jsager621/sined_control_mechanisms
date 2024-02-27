"""
Collection of message classes used in the simulation.
Each message class corresponds to one type of message.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable

"""
Message from central instance to participants
"""


@dataclass
class NewElecPriceMessage:
    price: float


@dataclass
class NewFeedinTariffMessage:
    tariff: float


@dataclass
class MaxNetFeedinMessage:
    max_feedin: float


@dataclass
class MaxNetLoadMessage:
    max_load: float


"""
Message from participants to central instance
"""


@dataclass
class LocalLoadMessage:
    load: float


@dataclass
class LocalFeedinMessage:
    feedin: float


@dataclass
class LocalVoltageMessage:
    voltage: float
