#!/bin/bash
for i in $(seq 10 11)
do
   rosrun stewart_platform PPO_Continuous.py --run $i
done

