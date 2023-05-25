import copy
import pickle
import threading
from typing import Any, List, Mapping
from fastapi import BackgroundTasks, FastAPI, Request
import sys
import json
from operator import itemgetter
import uvicorn
import asyncio
import httpx
import logging

from tinyfl.model import fedavg_models, create_model, test_model
from tinyfl.message import Register, StartRound, SubmitWeights

host: str
port: int

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s:     %(message)s - %(asctime)s",
)
logger = logging.getLogger(__name__)

client_lock = threading.Lock()
clients: List[str] = []


with open(sys.argv[1]) as f:
    config = json.load(f)
    host, port = itemgetter("host", "port")(config)

msg_id = 0

round_lock = threading.Lock()
round_id = 0

model_lock = threading.Lock()
model = create_model()

me = f"http://{host}:{port}"

consensus = 2
timeout = 1000
epochs = 3

quorum = threading.Condition()
client_models = []

# Generates the message ID, Returns the next message ID.
def next_msg_id() -> int:
    global msg_id
    ack_id = msg_id
    msg_id += 1
    return ack_id


app = FastAPI()

# Handles a GET request to root end point, Returns the response message
@app.get("/")
async def ping():
    return {
        "success": True,
        "message": "pong!",
        "me": me,
        "round": round_id,
        "clients": clients,
    }

# Handles a GET request to start New round, takes args backround tasks for starting a new round
# Returns a respose message
@app.get("/start_round")
async def start_round(background_tasks: BackgroundTasks):
    background_tasks.add_task(state_manager)
    return {"success": True}

# Handles a POST request to the root endpoint, takes args request, and background tasks for handling
# requests, Returns a response message
@app.post("/")
async def handle(req: Request, background_tasks: BackgroundTasks):
    msg = pickle.loads(await req.body())
    match msg:
        case Register(url=url):
            with client_lock:
                clients.append(url)
            return {"success": True, "message": "Registered"}
        case SubmitWeights(round=round, weights=weights):
            background_tasks.add_task(collect_weights, copy.deepcopy(weights))
            return {"success": True, "message": "Weights submitted"}
        case _:
            return {"success": False, "message": "Unknown message"}

# Manages the state of the learning porcess, waits for quorum of clients weights, 
# aggregates the models and then tests the aggregated models
def state_manager():
    r = asyncio.run(start_training())
    quorum_achieved: bool
    with quorum:
        logger.info("Waiting for quorum")
        quorum_achieved = quorum.wait(timeout)

        if not quorum_achieved:
            logger.error("Quorum not achieved!")
            return
        else:
            logger.info("Quorum achieved!")
            model.load_state_dict(fedavg_models(client_models))
            logger.info("Aggregated model")
            accuracy, loss = test_model(model)
            logger.info(f"Accuracy: {(accuracy):>0.1f}%, Loss: {loss:>8f}")

# Starts the training process by sending the start round message, returns a list of responsees from the clients
async def start_training():
    global round_id
    round_id += 1

    curr_weights = copy.deepcopy(model.state_dict())

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
                        )
                    ),
                )
                for party in clients
            ]
        )

# Collects the weights submitted by a client and add them to the list of clieeent models
# Args weights submitted by ther client
async def collect_weights(weights: Mapping[str, Any]):
    with round_lock:
        with quorum:
            client_models.append(weights)
            logger.info("Appended weights")

            if len(client_models) >= consensus:
                logger.info("Quorum notified")
                quorum.notify()

# Main, starts the server
def main():
    uvicorn.run(app, port=int(port), host=host)
