#!/bin/bash
export CARLA_ROOT=/home/group17/transfuser/carla
export WORK_DIR=/home/group17/transfuser
export TEAM_CONFIG=/home/group17/transfuser/model_ckpt/new_swin        # Change to your checkpoint config folder
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/team_code_transfuser
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/Swin_PTT_Files/submission_agent.py    # Change to your submission_agent.py path
export DEBUG_CHALLENGE=0
export RESUME=0
export DATAGEN=0
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/new_swin_37_41.json     # Change to your desired results output filename
export PORT=2000
export TM_PORT=8500
mkdir -p ${WORK_DIR}/results
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}