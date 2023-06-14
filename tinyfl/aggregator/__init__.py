import copy
import pickle
import threading
from typing import Any, Mapping
from fastapi import BackgroundTasks, FastAPI, Request
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import json
from operator import itemgetter
import uvicorn
import asyncio
import httpx
import logging

from tinyfl.model import create_model, test_model, stratified_split_dataset, strategies
from tinyfl.message import DeRegister, Register, StartRound, SubmitWeights

batch_size = 64

trainset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

testset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
testloader = DataLoader(testset, batch_size=batch_size)

host: str
port: int

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s:     %(message)s - %(asctime)s",
)
logger = logging.getLogger(__name__)

client_lock = threading.Lock()
clients = set()


with open(sys.argv[1]) as f:
    config = json.load(f)
    host, port, consensus, timeout, epochs, strategy = itemgetter(
        "host", "port", "consensus", "timeout", "epochs", "strategy"
    )(config)
    if strategies.get(strategy) is None:
        raise ValueError("Invalid aggregation model")
    strategy = strategies[strategy]

logger.info(f"{host}:{port} loaded from config.")
logger.info(f"Consensus: {consensus}")
logger.info(f"Timeout: {timeout}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Aggregation model: {strategy.__name__}")

msg_id = 0

round_lock = threading.Lock()
round_id = 0

model_lock = threading.Lock()
model = create_model()

me = f"http://{host}:{port}"


quorum = threading.Condition()

clients_models_lock = threading.Lock()
client_models = []


def next_msg_id() -> int:
    global msg_id
    ack_id = msg_id
    msg_id += 1
    return ack_id


app = FastAPI()


@app.get("/")
async def ping():
    return {
        "success": True,
        "message": "pong!",
        "me": me,
        "round": round_id,
        "clients": clients,
    }


@app.get("/start_round")
async def start_round(background_tasks: BackgroundTasks):
    background_tasks.add_task(state_manager)
    return {"success": True}


@app.post("/")
async def handle(req: Request, background_tasks: BackgroundTasks):
    msg = pickle.loads(await req.body())
    match msg:
        case Register(url=url):
            with client_lock:
                clients.add(url)
            logger.info(f"Client {url} registered")
            return {"success": True, "message": "Registered"}
        case SubmitWeights(round=round, weights=weights):
            background_tasks.add_task(collect_weights, copy.deepcopy(weights))
            return {"success": True, "message": "Weights submitted"}
        case DeRegister(url=url):
            with client_lock:
                clients.remove(url)
            logger.info(f"Client {url} de-registered")
            return {"success": True, "message": "De-registered"}
        case _:
            return {"success": False, "message": "Unknown message"}


def state_manager():
    global client_models
    with clients_models_lock:
        client_models = []
    asyncio.run(start_training())
    quorum_achieved: bool
    with quorum:
        logger.info("Waiting for quorum")
        quorum_achieved = quorum.wait(timeout)

        if not quorum_achieved:
            logger.error("Quorum not achieved!")
            return
        else:
            logger.info("Quorum achieved!")
            with clients_models_lock:
                model.load_state_dict(strategy(client_models))
            logger.info("Aggregated model")
            accuracy, loss = test_model(model, testloader)
            logger.info(f"Accuracy: {(accuracy):>0.1f}%, Loss: {loss:>8f}")


async def start_training():
    global round_id
    round_id += 1

    curr_weights = copy.deepcopy(model.state_dict())
    client_indices = stratified_split_dataset(trainset, len(clients))

    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *[
                client.post(
                    party,
                    data=pickle.dumps(
                        StartRound(
                            msg_id=next_msg_id(),
                            round=round_id,
                            epochs=epochs,
                            weights=curr_weights,
                            indices=indices,
                        )
                    ),
                )
                for party, indices in zip(clients, client_indices)
            ]
        )


async def collect_weights(weights: Mapping[str, Any]):
    with round_lock:
        with quorum:
            with clients_models_lock:
                notify_quorum = False
                if len(client_models) < consensus:
                    client_models.append(weights)
                    logger.info("Appended weights")
                    notify_quorum = len(client_models) == consensus
                if notify_quorum:
                    quorum.notify()


def main():
    uvicorn.run(app, port=int(port), host=host)
