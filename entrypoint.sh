#!/bin/bash
if [[ -z $CLIENT ]]; then
    echo "CLIENT environment variable not set"
    exit 1
fi
if [[ -z $TINYFL_CONFIG ]]; then
    echo "TINYFL_CONFIG environment variable not set"
    exit 1
fi
if [[ $CLIENT -eq 0 ]]; then
    echo "Starting aggregator"
    poetry run agg $TINYFL_CONFIG
    exit 0
elif [[ $CLIENT -eq 1 ]]; then
    echo "Starting party"
    poetry run party $TINYFL_CONFIG
    exit 0
else
    echo "CLIENT environment variable must be 0 or 1"
    exit 1
fi
