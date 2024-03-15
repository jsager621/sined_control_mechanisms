"""
Collection of message classes used in the simulation.
Each message class corresponds to one type of message.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable
import numpy as np
from mango.messages.codecs import json_serializable
from mango.messages.codecs import JSON


def get_codec():
    codec = JSON()

    codec.add_serializer(*LocalResidualScheduleMessage.__serializer__())

    codec.add_serializer(*TimeStepMessage.__serializer__())
    codec.add_serializer(*TimeStepReply.__serializer__())

    codec.add_serializer(*RegistrationMessage.__serializer__())
    codec.add_serializer(*RegistrationReply.__serializer__())

    codec.add_serializer(*ControlMechanismMessage.__serializer__())

    return codec


@dataclass
class AgentAddress:
    """
    Dataclass for sending and receiving Agent Information
    """

    host: str
    port: int
    agent_id: str

    def __eq__(self, other):
        return (
            self.host == other.host
            and self.port == other.port
            and self.agent_id == other.agent_id
        )

    def __hash__(self):
        return hash(("host", self.host, "port", self.port, "agent_id", self.agent_id))


"""
Message from central instance to participants
"""


@json_serializable
@dataclass
class RegistrationReply:
    ack: bool


# each field is an array of 96 values
# one setpoint for each 15 minute interval in the day-ahead schedule
@json_serializable
@dataclass
class ControlMechanismMessage:
    timestamp: int = field(default=None)
    tariff_adj: np.ndarray[float] = field(default=None)
    p_max: np.ndarray[float] = field(default=None)
    p_min: np.ndarray[float] = field(default=None)
    peak_price_dem: float = field(default=None)
    # demand peak price: to be effective instead of keeping energy and to not use grid energy
    #     needs to be lower than electricity price and higher than feedin tariff (times resolution)
    peak_price_gen: float = field(default=None)
    # generation peak price needs to be positive, height ist irrelevant


"""
Message from participants to central instance
"""


@json_serializable
@dataclass
class LocalResidualScheduleMessage:
    timestamp: int
    residual_schedule: np.ndarray[float]


@json_serializable
@dataclass
class RegistrationMessage:
    pass


"""
Simulation time sync messages
"""


@json_serializable
@dataclass
class TimeStepMessage:
    time: int
    c_id: int


@json_serializable
@dataclass
class TimeStepReply:
    c_id: int
