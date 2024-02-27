"""
Collection of message classes used in the simulation.
Each message class corresponds to one type of message.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable
from mango.messages.codecs import json_serializable
from mango.messages.codecs import JSON


def get_codec():
    codec = JSON()

    codec.add_serializer(*NewElecPriceMessage.__serializer__())
    codec.add_serializer(*NewFeedinTariffMessage.__serializer__())
    codec.add_serializer(*MaxNetFeedinMessage.__serializer__())
    codec.add_serializer(*MaxNetLoadMessage.__serializer__())

    codec.add_serializer(*LocalLoadMessage.__serializer__())
    codec.add_serializer(*LocalFeedinMessage.__serializer__())
    codec.add_serializer(*LocalVoltageMessage.__serializer__())

    codec.add_serializer(*TimeStepMessage.__serializer__())
    codec.add_serializer(*TimeStepReply.__serializer__())

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
class NewElecPriceMessage:
    price: float


@json_serializable
@dataclass
class NewFeedinTariffMessage:
    tariff: float


@json_serializable
@dataclass
class MaxNetFeedinMessage:
    max_feedin: float


@json_serializable
@dataclass
class MaxNetLoadMessage:
    max_load: float


"""
Message from participants to central instance
"""


@json_serializable
@dataclass
class LocalLoadMessage:
    load: float


@json_serializable
@dataclass
class LocalFeedinMessage:
    feedin: float


@json_serializable
@dataclass
class LocalVoltageMessage:
    voltage: float


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
