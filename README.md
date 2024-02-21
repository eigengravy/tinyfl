# tinyfl

a tiny federated learning framework built with pytorch and fastapi.

## requirements

- python
- torch
- torchvision
- httpx
- uvicorn
- fastapi

## installation

use poetry

```sh
poetry env use 3.11
poetry install
```

or

use conda

```sh
conda create -n tinyfl python=3.11
conda activate tinyfl
pip install torch torchvision httpx uvicorn fastapi
```

## quickstart

run the aggregator

```sh
poetry run agg configs/agg.config.json
```

run the parties

```sh
poetry run party configs/party0.config.json
poetry run party configs/party1.config.json
poetry run party configs/party2.config.json
```

get aggregator status

```sh
curl {aggregator}/
```

start training round

```sh
curl {aggregator}/start_round
```

## docker

```sh
docker build -t tinyfl .
docker run --name agg    --network=host -e CLIENT=0 -e TINYFL_CONFIG=configs/agg.config.json tinyfl:latest
docker run --name party0 --network=host -e CLIENT=1 -e TINYFL_CONFIG=configs/party0.config.json tinyfl:latest
docker run --name party1 --network=host -e CLIENT=1 -e TINYFL_CONFIG=configs/party1.config.json tinyfl:latest
docker run --name party2 --network=host -e CLIENT=1 -e TINYFL_CONFIG=configs/party2.config.json tinyfl:latest
```
