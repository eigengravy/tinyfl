import copy
import pickle
import threading
from typing import Any, Mapping
from fastapi import BackgroundTasks, FastAPI, Request
from torch.utils.data import DataLoader
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
from tinyfl.message import DeRegister, Register, StartSuperRound, SubmitSuperWeights, SubmitWeights

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
aggs = set()


with open(sys.argv[1]) as f:
    config = json.load(f)
    host, port, consensus, timeout, strategy, model, split = itemgetter(
        "host", "port", "consensus", "timeout", "strategy", "model", "split"
    )(config)
    if strategies.get(strategy) is None:
        raise ValueError("Invalid aggregation model")
    strategy = strategies[strategy]
    split_dataset = splits[split]

logger.info(f"{host}:{port} loa`ded from config.")
logger.info(f"Consensus: {consensus}")
logger.info(f"Timeout: {timeout}")
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
client_len = dict()


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
        "clients": aggs,
    }


@app.get("/start_super_round")
async def start_round(background_tasks: BackgroundTasks):
    background_tasks.add_task(state_manager)
    return {"success": True}


@app.post("/")
async def handle(req: Request, background_tasks: BackgroundTasks):
    msg = pickle.loads(await req.body())
    match msg:
        case Register(url=url):
            with client_lock:
                aggs.add(url)
            logger.info(f"Aggregator {url} registered")
            return {"success": True, "message": "Registered"}
        case SubmitSuperWeights(url=url, weights=weights):
            print("Received super weights")
            background_tasks.add_task(collect_weights, url, copy.deepcopy(weights))
            return {"success": True, "message": "Weights submitted"}
        case DeRegister(url=url, id=id):
            with client_lock:
                aggs.remove(url)
            logger.info(f"Aggregator {url} de-registered")
            return {"success": True, "message": "De-registered"}
        case _:
            return {"success": False, "message": "Unknown message"}


def state_manager():
    global client_models
    with clients_models_lock:
        client_models = dict()
    for client in aggs:
        client_models[client] = None
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
            # TODO: stop training after aggregation
            # asyncio.run(stop_training())
            with clients_models_lock:
                model.load_state_dict(
                    strategy(list(filter(lambda x: x != None, client_models.values())))
                )
            logger.info("Aggregated model")
            accuracy, loss = model.test_model(testloader)
            logger.info(f"Accuracy: {(accuracy):>0.1f}%, Loss: {loss:>8f}")


async def start_training():
    global round_id
    round_id += 1

    curr_weights = copy.deepcopy(model.state_dict())

    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(
            *[
                client.get(
                    party + "/len_clients",
                )
                for party in aggs
            ]
        )
        for i in responses:
            client_len[str(i.url).split("/len_clients")[0]] = int(
                i.json()["len_clients"]
            )

    client_indices = split_dataset(trainset, sum(client_len.values()))
    agg_indice = []
    cur = 0
    for agg in aggs:
        agg_indice.append((agg, client_indices[cur : cur + client_len[agg]]))
        cur += client_len[agg]

    # Tested to see if the indices are correct
    # print(client_len, list(map(lambda x: (x[0], len(x[1])), agg_indice)))

    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *[
                client.post(
                    party,
                    data=pickle.dumps(
                        StartSuperRound(
                            msg_id=next_msg_id(),
                            weights=curr_weights,
                            indices=indices,
                        )
                    ),
                )
                for party, indices in agg_indice
            ]
        )


async def collect_weights(url: str, weights: Mapping[str, Any]):
    with round_lock:
        with quorum:
            with clients_models_lock:
                notify_quorum = False
                models_submitted = len(
                    list(filter(lambda x: x != None, client_models.values()))
                )
                print("Models submitted:", models_submitted)
                if models_submitted < consensus:
                    client_models[url] = weights
                    logger.info("Appended weights")
                    notify_quorum = (models_submitted + 1) == consensus
                if notify_quorum:
                    quorum.notify()


def main():
    uvicorn.run(app, port=int(port), host=host)
