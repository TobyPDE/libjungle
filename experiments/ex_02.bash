#!/bin/bash
# 
# File:   ex_02.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 02: Performance of ensembles of DAGs w.r.t. their depth and the"
echo "number of DAGs. We train the DAGs without bagging. The same experiment is"
echo "repeated in ex_03 with bagging."
echo ""
echo "Experiment 02_01"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d  > "logs/ex_02_01_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d  > "logs/ex_02_01_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d  > "logs/ex_02_01_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d  > "logs/ex_02_01_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d  > "logs/ex_02_01_05.txt" 
wait
echo "Experiment 02_02"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_02_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d  > "logs/ex_02_02_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_02_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d  > "logs/ex_02_02_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_02_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d  > "logs/ex_02_02_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_02_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d  > "logs/ex_02_02_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_02_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d  > "logs/ex_02_02_05.txt" 
wait
echo "Experiment 02_03"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_03_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d  > "logs/ex_02_03_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_03_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d  > "logs/ex_02_03_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_03_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d  > "logs/ex_02_03_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_03_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d  > "logs/ex_02_03_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_03_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d  > "logs/ex_02_03_05.txt" 
wait
echo "Experiment 02_04"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_04_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d  > "logs/ex_02_04_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_04_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d  > "logs/ex_02_04_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_04_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d  > "logs/ex_02_04_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_04_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d  > "logs/ex_02_04_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_04_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d  > "logs/ex_02_04_05.txt" 
wait
echo "Experiment 02_05"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_05_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d  > "logs/ex_02_05_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_05_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d  > "logs/ex_02_05_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_05_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d  > "logs/ex_02_05_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_05_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d  > "logs/ex_02_05_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_05_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d  > "logs/ex_02_05_05.txt" 
wait
echo "Experiment 02_06"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_06_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d  > "logs/ex_02_06_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_06_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d  > "logs/ex_02_06_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_06_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d  > "logs/ex_02_06_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_06_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d  > "logs/ex_02_06_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_06_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d  > "logs/ex_02_06_05.txt" 
wait
echo "Experiment 02_07"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_07_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d  > "logs/ex_02_07_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_07_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d  > "logs/ex_02_07_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_07_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d  > "logs/ex_02_07_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_07_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d  > "logs/ex_02_07_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_07_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d  > "logs/ex_02_07_05.txt" 
wait
echo "Experiment 02_08"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_08_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d  > "logs/ex_02_08_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_08_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d  > "logs/ex_02_08_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_08_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d  > "logs/ex_02_08_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_08_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d  > "logs/ex_02_08_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_02_08_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d  > "logs/ex_02_08_05.txt" 
wait
