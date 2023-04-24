# tinyfl

a tiny federated learning framework built with pytorch and fastapi.

## requirements

- python==3.10.10
- torch==1.13.0
- torchvision==0.14.0
- httpx
- uvicorn
- fastapi

## installation

use poetry

```
poetry env use 3.10
poetry install
```

or

use conda

```
conda create -n tinyfl python=3.10.10
conda activate tinyfl
pip install torch==1.13.0 torchvision==0.14.0 httpx uvicorn fastapi
```

## quickstart

run the aggregator

```
poetry run agg agg.config.json
```

run the parties

```
poetry run party party0.config.json
poetry run party party1.config.json
poetry run party party2.config.json
```

get aggregator status

```
curl {aggregator}/
```

start training round

```
curl {aggregator}/start_round
```
