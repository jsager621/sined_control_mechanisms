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

    codec.add_serializer(*NewElecPriceMessage.__serializer__())
    codec.add_serializer(*NewFeedinTariffMessage.__serializer__())
    codec.add_serializer(*ControlResidualMessage.__serializer__())

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
Base Message to include timestep for the sent signal to indicate for which time step it is meant
"""


@json_serializable
@dataclass
class BaseMessage:
    timestep: int


"""
Message from central instance to participants
"""


@json_serializable
@dataclass
class NewElecPriceMessage(BaseMessage):
    price: float = field(default=None)


@json_serializable
@dataclass
class NewFeedinTariffMessage(BaseMessage):
    tariff: float = field(default=None)


@json_serializable
@dataclass
class ControlResidualMessage(BaseMessage):
    p_max: float = field(default=None)
    p_min: float = field(default=None)


@json_serializable
@dataclass
class RegistrationReply:
    ack: bool


# each field is an array of 96 values
# one setpoint for each 15 minute interval in the day-ahead schedule
@json_serializable
@dataclass
class ControlMechanismMessage:
    timestamp: int
    tariff: np.ndarray[float]
    p_max: np.ndarray[float]
    p_min: np.ndarray[float]


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
