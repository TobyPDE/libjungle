#!/bin/bash
# 
# File:   ex_07.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 07: Performance of ensembles of DAGs"
echo ""
echo "Experiment 07_01"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_07_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=65 -M=8 -d  > "logs/ex_07_01_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_07_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=65 -M=8 -d  > "logs/ex_07_01_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_07_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=65 -M=8 -d  > "logs/ex_07_01_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_07_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=65 -M=8 -d  > "logs/ex_07_01_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_07_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=65 -M=8 -d  > "logs/ex_07_01_05.txt" 
wait

