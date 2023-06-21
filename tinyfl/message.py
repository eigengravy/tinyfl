from dataclasses import dataclass
from typing import Any, List, Mapping


@dataclass
class Message:
    msg_id: int


@dataclass
class Register(Message):
    url: str


@dataclass
class StartRound(Message):
    round: int
    epochs: int
    weights: Mapping[str, Any]
    indices: List[int]
    aggregator: str


@dataclass
class SubmitWeights(Message):
    url: str
    round: int
    weights: Mapping[str, Any]
    final: bool = False

# TODO: Implement this using celery
# @dataclass 
# class StopRound(Message):
#     pass
    


@dataclass
class DeRegister(Message):
    url: str
