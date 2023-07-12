import copy
import pickle
import threading
from typing import Any, List, Mapping
from fastapi import BackgroundTasks, FastAPI, Request
from contextlib import asynccontextmanager
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import json
from operator import itemgetter
import uvicorn
import asyncio
import httpx
import logging

from tinyfl.model import (
    models,
    splits,
    strategies,
)
from tinyfl.message import (
    DeRegister,
    Register,
    StartRound,
    StartSuperRound,
    SubmitSuperWeights,
    SubmitWeights,
)

batch_size = 64

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
    (
        host,
        port,
        consensus,
        timeout,
        epochs,
        strategy,
        model,
        split,
        super_aggregator,
    ) = itemgetter(
        "host",
        "port",
        "consensus",
        "timeout",
        "epochs",
        "strategy",
        "model",
        "split",
        "super_aggregator",
    )(
        config
    )
    if strategies.get(strategy) is None:
        raise ValueError("Invalid aggregation model")
    strategy_name = strategy
    strategy = strategies[strategy]
    split_dataset = splits[split]

logger.info(f"{host}:{port} loaded from config.")
logger.info(f"Consensus: {consensus}")
logger.info(f"Timeout: {timeout}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Aggregation model: {strategy.__name__}")

msg_id = 0

round_lock = threading.Lock()
round_id = 0

model_lock = threading.Lock()
model = models[model].create_model()
trainset, testset = model.create_datasets()
trainloader = DataLoader(trainset, batch_size=batch_size)
testloader = DataLoader(testset, batch_size=batch_size)

me = f"http://{host}:{port}"


quorum = threading.Condition()

clients_models_lock = threading.Lock()
client_models = dict()


def next_msg_id() -> int:
    global msg_id
    ack_id = msg_id
    msg_id += 1
    return ack_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model initialized.")
    r = httpx.post(
        super_aggregator, data=pickle.dumps(Register(msg_id=next_msg_id(), url=me))
    )
    yield
    logger.info("Shutting down")
    r = httpx.post(
        strategy_name, data=pickle.dumps(DeRegister(msg_id=next_msg_id(), url=me))
    )
    if r.status_code == 200:
        logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


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


@app.get("/len_clients")
async def get_len_clients():
    return {"success": True, "len_clients": len(clients)}


@app.post("/")
async def handle(req: Request, background_tasks: BackgroundTasks):
    msg = pickle.loads(await req.body())
    match msg:
        case Register(url=url):
            with client_lock:
                clients.add(url)
            logger.info(f"Client {url} registered")
            return {"success": True, "message": "Registered"}
        case SubmitWeights(round=round, weights=weights, url=url, final=final):
            background_tasks.add_task(
                collect_weights, url, copy.deepcopy(weights), final
            )
            return {"success": True, "message": "Weights submitted"}
        case DeRegister(url=url, id=id):
            with client_lock:
                clients.remove(url)
            logger.info(f"Client {url} de-registered")
            return {"success": True, "message": "De-registered"}
        case StartSuperRound(weights=weights, indices=indices):
            background_tasks.add_task(state_manager, weights, indices)
        case _:
            return {"success": False, "message": "Unknown message"}


def state_manager(weights: Any = None, indices: Any = None):
    global client_models
    with clients_models_lock:
        client_models = dict()
    for client in clients:
        client_models[client] = None
    asyncio.run(start_training(weights, indices))
    quorum_achieved: bool
    with quorum:
        logger.info("Waiting for quorum")
        quorum_achieved = quorum.wait(timeout)

        if not quorum_achieved and strategy_name != "fedprox":
            logger.error("Quorum not achieved!")
            return
        else:
            logger.info("Quorum achieved!")
            # TODO: stop training after aggregation
            # asyncio.run(stop_training())
            with clients_models_lock:
                model.load_state_dict(
                    strategy(list(filter(lambda x: x != None, client_models.values())))
                )
            logger.info("Aggregated model")
            accuracy, loss = model.test_model(testloader)
            logger.info(f"Accuracy: {(accuracy):>0.1f}%, Loss: {loss:>8f}")
            asyncio.run(submit_model())


async def submit_model():
    r = httpx.post(
        super_aggregator,
        data=pickle.dumps(
            SubmitSuperWeights(
                url=me,
                msg_id=next_msg_id(),
                weights=copy.deepcopy(model.state_dict()),
            )
        ),
    )
    if r.status_code == 200:
        logger.info("Model submitted")
    else:
        logger.error("Model submission failed")


#
# async def stop_training():
#     async with httpx.AsyncClient() as client:
#         return await asyncio.gather(
#             *[
#                 client.post(party, data=pickle.dumps(StopRound(msg_id=next_msg_id())))
#                 for party in clients
#             ]
#         )


async def start_training(weights: Any = None, indices: Any = None):
    global round_id
    round_id += 1

    if weights is not None:
        model.load_state_dict(weights)
    curr_weights = copy.deepcopy(model.state_dict())
    if indices is None:
        client_indices = split_dataset(trainset, len(clients))
    else:
        client_indices = indices

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
                            aggregator=strategy_name,
                        )
                    ),
                )
                for party, indices in zip(clients, client_indices)
            ]
        )


async def collect_weights(url: str, weights: Mapping[str, Any], final: bool):
    with round_lock:
        with quorum:
            with clients_models_lock:
                notify_quorum = False
                if strategy_name == "fedavg":
                    models_submitted = len(
                        list(filter(lambda x: x != None, client_models.values()))
                    )
                    if models_submitted < consensus:
                        client_models[url] = weights
                        logger.info("Appended weights")
                        notify_quorum = (models_submitted + 1) == consensus
                    if notify_quorum:
                        quorum.notify()
                elif strategy_name == "fedprox":
                    client_models[url] = weights
                    logger.info("Appended weights")


def main():
    uvicorn.run(app, port=int(port), host=host)
