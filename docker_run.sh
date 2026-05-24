#! /usr/bin/env bash
#docker run --gpus all -it groundedsam:stream bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --gpus all -it \
	-p 8091:8091 \
	-v "${SCRIPT_DIR}:/home/appuser/Grounded-SAM-2" \
	-w /home/appuser/Grounded-SAM-2 \
	groundedsam:latest bash
