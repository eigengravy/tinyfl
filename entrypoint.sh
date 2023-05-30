#!/bin/bash
if [[ -z $CLIENT ]]; then
    echo "CLIENT environment variable not set"
    exit 1
fi
if [[ $CLIENT -eq -1 ]]; then
    echo "Starting server"
    poetry run agg configs/agg.config.json
    exit 0
elif [[ $CLIENT -ge 0 && $CLIENT -le 2 ]]; then
    echo "Starting client $CLIENT"
    poetry run party configs/party$(echo $CLIENT).config.json
    exit 0
else 
    echo "CLIENT environment variable must be -1, 0, 1, or 2"
    exit 1
fi

