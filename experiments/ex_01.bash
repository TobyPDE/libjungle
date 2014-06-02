#!/bin/bash
# 
# File:   ex_01.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 01: Influence of the iteration limit on the number of nodes"
echo "and the generalization error."
echo ""
echo "Experiment 01_01"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=5 -d  > "logs/ex_01_01_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=5 -d  > "logs/ex_01_01_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=5 -d  > "logs/ex_01_01_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=5 -d  > "logs/ex_01_01_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=5 -d  > "logs/ex_01_01_05.txt" &
wait
echo "Experiment 01_02"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_02_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=10 -d  > "logs/ex_01_02_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_02_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=10 -d  > "logs/ex_01_02_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_02_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=10 -d  > "logs/ex_01_02_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_02_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=10 -d  > "logs/ex_01_02_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_02_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=10 -d  > "logs/ex_01_02_05.txt" &
wait
echo "Experiment 01_03"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_03_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=15 -d  > "logs/ex_01_03_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_03_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=15 -d  > "logs/ex_01_03_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_03_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=15 -d  > "logs/ex_01_03_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_03_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=15 -d  > "logs/ex_01_03_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_03_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=15 -d  > "logs/ex_01_03_05.txt" &
wait
echo "Experiment 01_04"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_04_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=20 -d  > "logs/ex_01_04_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_04_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=20 -d  > "logs/ex_01_04_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_04_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=20 -d  > "logs/ex_01_04_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_04_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=20 -d  > "logs/ex_01_04_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_04_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=20 -d  > "logs/ex_01_04_05.txt" &
wait
echo "Experiment 01_05"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_05_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=25 -d  > "logs/ex_01_05_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_05_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=25 -d  > "logs/ex_01_05_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_05_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=25 -d  > "logs/ex_01_05_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_05_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=25 -d  > "logs/ex_01_05_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_05_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=25 -d  > "logs/ex_01_05_05.txt" &
wait
echo "Experiment 01_06"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_06_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=30 -d  > "logs/ex_01_06_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_06_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=30 -d  > "logs/ex_01_06_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_06_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=30 -d  > "logs/ex_01_06_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_06_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=30 -d  > "logs/ex_01_06_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_06_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=30 -d  > "logs/ex_01_06_05.txt" &
wait
echo "Experiment 01_07"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_07_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=35 -d  > "logs/ex_01_07_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_07_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=35 -d  > "logs/ex_01_07_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_07_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=35 -d  > "logs/ex_01_07_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_07_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=35 -d  > "logs/ex_01_07_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_07_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=35 -d  > "logs/ex_01_07_05.txt" &
wait
echo "Experiment 01_08"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_08_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=40 -d  > "logs/ex_01_08_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_08_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=40 -d  > "logs/ex_01_08_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_08_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=40 -d  > "logs/ex_01_08_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_08_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=40 -d  > "logs/ex_01_08_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_08_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=40 -d  > "logs/ex_01_08_05.txt" &
wait
echo "Experiment 01_09"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_09_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=45 -d  > "logs/ex_01_09_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_09_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=45 -d  > "logs/ex_01_09_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_09_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=45 -d  > "logs/ex_01_09_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_09_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=45 -d  > "logs/ex_01_09_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_09_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=45 -d  > "logs/ex_01_09_05.txt" &
wait
echo "Experiment 01_10"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_10_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=50 -d  > "logs/ex_01_10_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_10_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=50 -d  > "logs/ex_01_10_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_10_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=50 -d  > "logs/ex_01_10_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_10_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=50 -d  > "logs/ex_01_10_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_10_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=50 -d  > "logs/ex_01_10_05.txt" &
wait
echo "Experiment 01_11"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_11_01.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=55 -d  > "logs/ex_01_11_01.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_11_02.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=55 -d  > "logs/ex_01_11_02.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_11_03.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=55 -d  > "logs/ex_01_11_03.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_11_04.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=55 -d  > "logs/ex_01_11_04.txt" &
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_01_11_05.txt" -V=../training_data/test_decision_jungle.txt -v=3 -I=55 -d  > "logs/ex_01_11_05.txt" &
