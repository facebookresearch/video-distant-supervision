# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# A script with a list of commands for submitting SLURM jobs

#Pretraining ablation
JOB_NAME=ablation_distribution_matching_top3
python tools/submit.py --cfg configs/HowTo100M/ablation_distribution_matching_top3.yaml --job_dir  ./${JOB_NAME}/   --num_shards 4  --comment "ablation_distribution_matching_top3" --name ${JOB_NAME} --use_volta32 

#Pretraining stage 1
JOB_NAME=distribution_matching_top3
python tools/submit.py --cfg configs/HowTo100M/distribution_matching_top3.yaml --job_dir  ./${JOB_NAME}/   --num_shards 16  --comment "distribution_matching_top3" --name ${JOB_NAME} --use_volta32 --timeout 2500

#Pretraining stage 2
JOB_NAME=distribution_matching_top3_c30
python tools/submit.py --cfg configs/HowTo100M/distribution_matching_top3_c30.yaml --job_dir  ./${JOB_NAME}/   --num_shards 16  --comment "distribution_matching_top3_c30" --name ${JOB_NAME} --use_volta32 --timeout 2500

#step forecasting on COIN
JOB_NAME=step_forecasting
python tools/submit.py --cfg configs/COIN/step_forecasting.yaml --job_dir  ./${JOB_NAME}/   --num_shards 4 --comment "step_forecasting" --name ${JOB_NAME} --use_volta32  --timeout 1200

#recognition of procedural activity on COIN
JOB_NAME=recognition_procedural_activity
python tools/submit.py --cfg configs/COIN/recognition_procedural_activity.yaml --job_dir  ./${JOB_NAME}/   --num_shards 4  --comment "recognition_procedural_activity" --name ${JOB_NAME} --use_volta32  --timeout 1200

# Epic-Kitchens-100
JOB_NAME=egocentric_activity_classification
python tools/submit.py --cfg configs/EK/egocentric_activity_classification.yaml --job_dir  ./${JOB_NAME}/   --num_shards 8  --comment "egocentric_activity_classification" --name ${JOB_NAME} --use_volta32 --timeout 1200