#!/bin/bash
# 
# File:   ex_05.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 05: Performance of ensembles of large DAGs without bagging"
echo ""
echo "Experiment 05_01"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_05_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=150 -d  > "logs/ex_05_01_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_05_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=150 -d  > "logs/ex_05_01_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_05_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=150 -d  > "logs/ex_05_01_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_05_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=150 -d  > "logs/ex_05_01_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_05_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=150 -d  > "logs/ex_05_01_05.txt" 
wait
