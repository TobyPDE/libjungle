#!/bin/bash
# 
# File:   ex_08.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 08: Influence of parent node sorting"
echo ""
echo "Experiment 08_01"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_05.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_06.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_06.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_07.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_07.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_08.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_08.txt" &
wait
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_09.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_09.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_01_10.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=0 -d > "logs/ex_08_01_10.txt" &
echo "Experiment 08_02"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_05.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_06.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_06.txt" &
wait
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_07.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_07.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_08.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_08.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_09.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_09.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_08_02_10.txt" -V=../training_data/test_decision_jungle.txt -v=3 -P=1 -d > "logs/ex_08_02_10.txt" &

