# LibJungle
This is a basic implementation of decision jungles for multi class classification as proposed by Shotton et al. in [1]. 
It implements the LSearch training algorithm with the optimizations discussed in [2]. 

## Compile code
If you want to use the tool, you have to compile it from source. You need to have
the following tools installed
*   gcc >= 4.8 (C++ 11 compliant)
*   CMake >= 2.6
*   Boost >= 1.40.0 (Filesystem and Tokenizer)
*   openMP (no specific version. Only very basic features are required)
Execute the following commands in your shell in order to download and compile the
code:
```shell
$ hg clone https://bitbucket.org/geekStack/libjungle
# Change to the code directory
$ cd libjungle
# We don't want to have objects files mixed with source files in one directory
# Hence, we compile the project in a build directory
$ mkdir build
$ cd build
# Create the makefile using CMake
$ cmake ../
# Compile the library
$ make
# Test if everything went as expected
$ ./jungle help
```
If everything went as expected, the last command should output the help dialog. 

## Training sets
Data is provided in the form of CSV files. Training sets have the 
following structure:
```
[class label],[feature 1],...,[feature n]
```
*Important*: The class labels must start at 0 and the features are real numbers. 

## Usage
You can use the tool for learning decision jungles as well as classifying new data
based on an already learned model. Please see '''./jungle help [command]''' for 
detailed information about the arguments individual commands accept. 

### Learn a jungle
You can use the 'learn' command in order to learn a new jungle. The following
command will learn a decision jungle of 8 DAGs from the file `training_set.txt`
and will save the model to the file `model.txt`:
```shell
$ ./jungle train training_set.txt model.txt -M=8
``` 
Please see `jungle help train` for further information.

### Test performance
You can use the 'classify' command in order to evaluate the performance of a 
learned decision jungle on a new training set. The following command will output
the test error of the decision jungle stored in `model.txt` on the test set
`test_set.txt`:
```shell
$ ./jungle classify test_set.txt model.txt
``` 

## License and Copyright
Everything except sse.h and fastlog.h is licensed under the following BSD license:

Copyright (c) 2014, Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
- The names of its contributors may not be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TOBIAS POHLEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Fast log2 approximation (sse.h, fastlog.h) Copyright (C) 2011 Paul Mineiro.
Further information under: https://code.google.com/p/fastapprox/


## Miscellaneous
Please cite the code as follows, if you use it in a publication:
```
@misc{TPDJ, 
   author = {Tobias Pohlen}, 
   title = {LibJungle - Decision Jungle Library}, 
   howpublished = {\url{https://bitbucket.org/geekStack/libjungle}} 
} 
```

## References
- [1] Jamie Shotton, Toby Sharp, Pushmeet Kohli, Sebastian Nowozin, John Winn, and Antonio Criminisi, "Decision jungles: Compact and rich models for classification". In C.J.C. Burges, L. Botou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems 26, pages 234-242. Curran Associates, Inc., 2013.
- [2] Tobias Pohlen, "Decision Jungles". RWTH Aachen University. Aachen, Germany.