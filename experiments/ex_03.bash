#!/bin/bash
# 
# File:   ex_03.bash
# Author: pohlen
#
# Created on Jun 2, 2014, 11:23:32 AM
#
echo "Experiment 03: Performance of ensembles of DAGs w.r.t. their depth and the"
echo "number of DAGs. We train the DAGs with bagging. The same experiment is"
echo "repeated in ex_02 without bagging."
echo ""
echo "Experiment 03_01"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_01_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d -B > "logs/ex_03_01_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_01_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d -B > "logs/ex_03_01_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_01_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d -B > "logs/ex_03_01_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_01_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d -B > "logs/ex_03_01_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_01_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=10 -M=30 -d -B > "logs/ex_03_01_05.txt" 
#wait
#echo "Experiment 03_02"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_02_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d -B > "logs/ex_03_02_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_02_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d -B > "logs/ex_03_02_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_02_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d -B > "logs/ex_03_02_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_02_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d -B > "logs/ex_03_02_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_02_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=15 -M=30 -d -B > "logs/ex_03_02_05.txt" 
#wait
#echo "Experiment 03_03"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_03_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d -B > "logs/ex_03_03_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_03_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d -B > "logs/ex_03_03_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_03_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d -B > "logs/ex_03_03_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_03_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d -B > "logs/ex_03_03_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_03_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=20 -M=30 -d -B > "logs/ex_03_03_05.txt" 
#wait
#echo "Experiment 03_04"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_04_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d -B > "logs/ex_03_04_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_04_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d -B > "logs/ex_03_04_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_04_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d -B > "logs/ex_03_04_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_04_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d -B > "logs/ex_03_04_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_04_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=25 -M=30 -d -B > "logs/ex_03_04_05.txt" 
#wait
#echo "Experiment 03_05"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_05_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d -B > "logs/ex_03_05_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_05_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d -B > "logs/ex_03_05_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_05_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d -B > "logs/ex_03_05_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_05_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d -B > "logs/ex_03_05_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_05_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=30 -M=30 -d -B > "logs/ex_03_05_05.txt" 
#wait
#echo "Experiment 03_06"
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_06_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d -B > "logs/ex_03_06_01.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_06_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d -B > "logs/ex_03_06_02.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_06_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d -B > "logs/ex_03_06_03.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_06_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d -B > "logs/ex_03_06_04.txt" 
#../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_06_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=35 -M=30 -d -B > "logs/ex_03_06_05.txt" 
#wait
echo "Experiment 03_07"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_07_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d -B > "logs/ex_03_07_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_07_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d -B > "logs/ex_03_07_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_07_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d -B > "logs/ex_03_07_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_07_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d -B > "logs/ex_03_07_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_07_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=40 -M=30 -d -B > "logs/ex_03_07_05.txt" 
wait
echo "Experiment 03_08"
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_08_01.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d -B > "logs/ex_03_08_01.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_08_02.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d -B > "logs/ex_03_08_02.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_08_03.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d -B > "logs/ex_03_08_03.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_08_04.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d -B > "logs/ex_03_08_04.txt" 
../build/jungle train "../training_data/train_decision_jungle.txt" "models/ex_03_08_05.txt" -V=../training_data/test_decision_jungle.txt -v=2 -I=35 -D=45 -M=30 -d -B > "logs/ex_03_08_05.txt" 
wait
