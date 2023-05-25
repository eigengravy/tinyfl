from dataclasses import dataclass
from typing import Any, List, Mapping

# Base class for messages used in the learning proccess
@dataclass
class Message:
    msg_id: int

# Message sent by a client to register with the server
@dataclass
class Register(Message):
    url: str

# Message sent by the server to start a new training round
# Args: round: the round ID, epochs: The number of training rounds, wieghts: the intitial weights of each
# round
@dataclass
class StartRound(Message):
    round: int
    epochs: int
    weights: Mapping[str, Any]

# Message sent by a client to submit their weights
# Args: round, weights
@dataclass
class SubmitWeights(Message):
    round: int
    weights: Mapping[str, Any]
