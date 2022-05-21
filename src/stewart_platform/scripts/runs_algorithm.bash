#!/bin/bash
for i in $(seq 0 5)
do
   rosrun stewart_platform PPO_Continuous.py --run $i
done

