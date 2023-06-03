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


@dataclass
class SubmitWeights(Message):
    round: int
    weights: Mapping[str, Any]
