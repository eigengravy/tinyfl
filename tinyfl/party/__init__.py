import copy
from typing import List
from fastapi import BackgroundTasks, FastAPI, Request
from contextlib import asynccontextmanager
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import uvicorn
import sys
import json
from operator import itemgetter
import logging
import httpx
import pickle

from tinyfl.model import create_model, test_model, train_model, subset_from_indices
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
aggregator: str

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s:     %(message)s - %(asctime)s",
)
logger = logging.getLogger(__name__)

with open(sys.argv[1]) as f:
    config = json.load(f)
    host, port, aggregator = itemgetter("host", "port", "aggregator")(config)

logger.info(f"{host}:{port} loaded from config.")

me = f"http://{host}:{port}"
msg_id = 0
round_id = 0
model = create_model()


def next_msg_id() -> int:
    global msg_id
    ack_id = msg_id
    msg_id += 1
    return ack_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model initialized.")
    r = httpx.post(
        aggregator, data=pickle.dumps(Register(msg_id=next_msg_id(), url=me))
    )
    yield
    logger.info("Shutting down")
    r = httpx.post(
        aggregator, data=pickle.dumps(DeRegister(msg_id=next_msg_id(), url=me))
    )
    if r.status_code == 200:
        logger.info("Shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def ping():
    return {"success": True, "message": "pong!", "me": me, "round": round_id}


@app.post("/")
async def handle(req: Request, background_tasks: BackgroundTasks):
    msg = pickle.loads(await req.body())
    match msg:
        case StartRound(round=round, epochs=epochs, weights=weights, indices=indices):
            background_tasks.add_task(run_training, weights, epochs, round_id, indices)
            return {"success": True, "message": f"Starting round {round_id}"}
        case _:
            return {"success": False, "message": "Unknown message"}


def run_training(weights, epochs: int, round: int, indices: List[int]):
    global round_id
    round_id = round

    model.load_state_dict(weights)

    train_subset = subset_from_indices(trainset, indices)
    trainloader = DataLoader(train_subset, num_workers=4, batch_size=batch_size)

    logger.info("Training started")
    train_model(model=model, epochs=epochs, trainloader=trainloader)
    logger.info("Training ended")

    r = httpx.post(
        aggregator,
        data=pickle.dumps(
            SubmitWeights(
                msg_id=next_msg_id(),
                round=round,
                weights=copy.deepcopy(model.state_dict()),
            )
        ),
    )

    if r.status_code == 200:
        logger.info("Submitted weights")


def run_testing():
    accuracy, loss = test_model(model, testloader)
    logger.info(f"Accuracy: {(accuracy):>0.1f}%, Loss: {loss:>8f}")


def main():
    uvicorn.run(app, port=int(port), host=host)
