#!/bin/bash
for i in $(seq 0 10)
do
   rosrun stewart_platform DDPG_Continuous_revision.py --run $i
done

