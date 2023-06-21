# Computational-BiologyEx3

This exercise has been writen by Roi Avraham and Dina Englard.
# Description
For this assignment we build neural networks with the help of a genetic algorithm that will 
learn the patterns and be able to predict whether a certain string matches the pattern. 
You can decide and determine the structure of the network in terms of the number of layers and the number of 
connections for each problem(nn0 and nn1).
<br>There are 4 exe files:
1) buildnet0.exe
2) buildnet1.exe
3) runnet0.exe
4) runnet1.exe

# The output files:
- buildnet0.exe output file named wnet0.txt. This file contains definitions of the network structure 
and the weights of the network for nn0.
- buildnet1.exe output file named wnet1.txt. This file contains definitions of the network structure 
and the weights of the network for nn1.
- runnet0.exe output file named classifications0.txt. This file Contains 20000 lines where each line contains
the appropriate classification (ie the number 0 or 1) for this string
- runnet1.exe output file named classifications1.txt. This file Contains 20000 lines where each line contains
the appropriate classification (ie the number 0 or 1) for this string

# Installation
In order to run buildnet0.exe, you will need to enter the next command in the terminal:
<br> buildnet0.exe {path of the train file} {path of the test file}
<br>
<br> In order to run buildnet1.exe, you will need to enter the next command in the terminal:
<br> buildnet1.exe {path of the train file} {path of the test file}
<br>
<br> In order to run runnet0.exe, you will need to enter the next command in the terminal:
<br> runnet0.exe {path of wnet0} {path of testnet0}
<br>
<br> In order to run runnet1.exe, you will need to enter the next command in the terminal:
<br> runnet1.exe {path of wnet1} {path of testnet1}
