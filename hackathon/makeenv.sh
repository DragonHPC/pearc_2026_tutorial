#!/bin/bash

. $PROJECT/../shared/bin/setup

python -m venv testenv
. testenv/bin/activate

module load openmpi

wget https://raw.githubusercontent.com/DragonHPC/pearc_2026_tutorial/refs/heads/main/.devcontainer/requirements.txt

pip install -r requirements.txt

dragon-config add --ucx-runtime-lib="/opt/packages/ucx/v1.20.0/gcc15.2.1-p20250906-x86-64-v3/lib64"

# add this bin path which has a "dragon" wrapper that works around a domain name incompatibility
export PATH=/ocean/projects/tra260009p/shared/bin:${PATH}

echo "Environment setup complete"
echo 'LLM is located in $PROJECT/../shared/model/SmolLM3_3B/'
