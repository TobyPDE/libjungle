/**
 * This file contains the command line interface for the decision jungle library. It essentially let's you train a model
 * on some dataset provided by a data file, save the model to a model file and predict new values from a data file
 * based on a learned model from a model file. Please see the help option for further information. 
 * 
 * @author Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
 * @version 1.0
 */

#include "cliInterface.h"
#include "jungle.h"
#include "jungleTrain.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <exception>
#include <boost/timer.hpp>


using namespace JunglePP;

/**
 * Register all available functions
 */
AbstractCLIFunction::Factory::ClassMap AbstractCLIFunction::Factory::functionMap;
AbstractCLIFunction::RegisterFunction<HelpCLIFunction> HelpCLIFunction::reg("help");
AbstractCLIFunction::RegisterFunction<ClassifyCLIFunction> ClassifyCLIFunction::reg("classify");
AbstractCLIFunction::RegisterFunction<TrainCLIFunction> TrainCLIFunction::reg("train");
AbstractCLIFunction::RegisterFunction<VersionCLIFunction> VersionCLIFunction::reg("version");

int main(int argc, const char** argv)
{
    try {
       // Parse the arguments into an argument bag
        ArgumentBag::ptr arguments = ArgumentBag::Factory::createFromCLIArguments(1, argc, argv);
        // Try to load the requested function
        AbstractCLIFunction::ptr function = AbstractCLIFunction::Factory::createFromArgumentBag(arguments);
        // Execute the requested function
        return function->execute();
    }
    catch (std::exception & e) {
        std::cout << "There was an error." << std::endl;
        std::cout << " -> " << e.what() << std::endl;
        std::cout << "Please see '$ jungle help' for more information." << std::endl;
        return 1;
    }
}

ArgumentBag::ptr ArgumentBag::Factory::createFromCLIArguments(const int start, const int argc, const char** argv)
{
    // Create a new instance 
    ArgumentBag::ptr result = ArgumentBag::Factory::create();
    
    // Parse the arguments
    for (int i = start; i < argc; i++)
    {
        // Is this a parameter or an argument?
        if (argv[i][0] == '-')
        {
            // This is a parameter
            // Check if there is an assignment (i.e. if there is a = sign)
            std::string parameter(argv[i]);
            
            std::string::size_type equalPos = parameter.find('=');
            
            if (equalPos == std::string::npos)
            {
                // There is no sign, add this with value "1"
                result->getParameters()[parameter.substr(1, parameter.size() - 1)] = "1";
            }
            else
            {
                // There is an assignment
                result->getParameters()[parameter.substr(1, equalPos - 1)] = parameter.substr(equalPos+1, parameter.size() - equalPos);
            }
        }
        else
        {
            // This is an argument, simply add it
            result->getArguments().push_back(argv[i]);
        }
    }
    
    return result;
}

AbstractCLIFunction::ptr AbstractCLIFunction::Factory::createFromArgumentBag(ArgumentBag::ptr _argumentBag) throw(CLIFunctionNotFoundException)
{
    // Is there any argument given?
    if (_argumentBag->getArguments().size() == 0)
    {
        throw CLIFunctionNotFoundException("No function was specified.");
    }
    
    // Get the name of the requested function and remove it from the argument stack
    std::string functionName = _argumentBag->getArguments().front();
    _argumentBag->getArguments().erase(_argumentBag->getArguments().begin());
    
    // Search for the function in the registered functions
    AbstractCLIFunction::ptr result = AbstractCLIFunction::Factory::createFromName(functionName);
    // Assign the argument bag
    result->arguments = _argumentBag;
    
    return result;
}

AbstractCLIFunction::ptr AbstractCLIFunction::Factory::createFromName(const std::string & _name) throw(CLIFunctionNotFoundException)
{
    // Search for the function in the registered functions
    ClassMap::iterator it = functionMap.find(_name);
    if (it == functionMap.end())
    {
        // The function name is unknown
        throw CLIFunctionNotFoundException("Unknown function name.");
    }
    
    AbstractCLIFunction::ptr result(it->second());
    return result;
}

std::vector<std::string> AbstractCLIFunction::Factory::getRegisteredNames()
{
    std::vector<std::string> result;
    
    // Create the key vector
    for (ClassMap::iterator it = functionMap.begin(); it != functionMap.end(); ++it)
    {
        result.push_back(it->first);
    }
    
    return result;
}

int HelpCLIFunction::displayGlobalHelp()
{
    // Display the the default help dialog
    // All functions are listed with their short text
    std::cout << "Decision Jungle Library" << std::endl << std::endl;
    std::cout << "List of commands:" << std::endl << std::endl;
    
    // Get all commands
    std::vector<std::string> commandNames = AbstractCLIFunction::Factory::getRegisteredNames();
    
    for (std::vector<std::string>::iterator it = commandNames.begin(); it != commandNames.end(); ++it)
    {
        // Create an instance of the current function in order to access the short help text
        AbstractCLIFunction::ptr currentFunction = AbstractCLIFunction::Factory::createFromName(*it);
        std::cout << ' ' << std::setw(15) << std::left << *it << ' ' << currentFunction->shortHelp() << std::endl;
    }
    std::cout << std::endl;
    std::cout << "For more information about a certain command call: " << std::endl;
    std::cout << " $ jungle help {command}" << std::endl; 
    return 0;
}

int HelpCLIFunction::displayFunctionHel(const std::string& _name)
{
    try {
        // Try to get an instance of this function
        AbstractCLIFunction::ptr function = AbstractCLIFunction::Factory::createFromName(_name);
        // Display the help dialog
        std::cout << function->help() << std::endl;
        return 0;
    }
    catch (CLIFunctionNotFoundException & e) {
        std::cout << "The requested function could not be found." << std::endl;
        return 1;
    }
}

int HelpCLIFunction::execute()
{
    // Is there a function requested?
    if (this->getArguments()->getArguments().size() > 0)
    {
        // Yes, displays the help of this function
        return this->displayFunctionHel(this->getArguments()->getArguments().front());
    }
    else
    {
        // Nope, display the global help
        return this->displayGlobalHelp();
    }
}

const char* HelpCLIFunction::help()
{
    return  "USAGE \n"
            " $ jungle help [command] \n\n"
            "PARAMETERS\n"
            " There are no parameters for this command\n\n"
            "DESCRIPTION\n"
            " This command displays either the help dialog for the entire\n"
            " library/cli interface or the help dialog for a specific\n"
            " command.\n";
}


const char* HelpCLIFunction::shortHelp()
{
    return "Displays either the global help dialog or the help for a specific command";
}


int ClassifyCLIFunction::execute()
{
    // There must be a model file and a training set
    if (getArguments()->getArguments().size() != 2)
    {
        std::cout << "Please use the command as follows:" << std::endl;
        std::cout << " $ jungle classify [parameters] {trainingset} {model}" << std::endl;
        std::cout << "See '$ jugle help classify' for more information." << std::endl;
        return 1;
    }
    
    // Load the jungle
    std::cout << "Loading jungle" << std::endl;
    Jungle::ptr jungle = Jungle::Factory::createFromFile(getArguments()->getArguments().at(1), true);
    
    // Load the training set
    std::cout << "Loading test set" << std::endl;
    TrainingSet::ptr testSet = TrainingSet::Factory::createFromFile(getArguments()->getArguments().at(0), true);
    
    std::cout << std::endl;
    
    // Display some error statistics
    TrainingStatistics::ptr statisticsTool = TrainingStatistics::Factory::create();
    
    std::cout << "Error: " << statisticsTool->trainingError(jungle, testSet) << std::endl;
    
    TrainingSet::freeTrainingExamples(testSet);
    
    return 0;
}


const char* ClassifyCLIFunction::help()
{
    return  "USAGE \n"
            " $ jungle classify [parameters] {traininset} {model} \n\n"
            "PARAMETERS\n"
            " There are no parameters for this command\n\n"
            "DESCRIPTION\n"
            " This command classifies known data (i.e. a training set).\n";
}


const char* ClassifyCLIFunction::shortHelp()
{
    return "Classifies known data (error statistics, confusion matrix)";
}

void TrainCLIFunction::loadParametersToTrainer(JungleTrainer::ptr _trainer)
{
    // Iterate over all parameters given and ignore unknown parameters
    std::map<std::string, std::string> parameters = getArguments()->getParameters();
    for (std::map<std::string, std::string>::iterator it = parameters.begin(); it != parameters.end(); ++it)
    {
        switch (ParameterConverter::getChar(it->first))
        {
            case 'M':
                _trainer->setNumDAGs(ParameterConverter::getInt(it->second));
                break;
                
            case 'N':
                _trainer->setNumTrainingSamples(ParameterConverter::getInt(it->second));
                break;
            
            case 'F':
                _trainer->setNumFeatureSamples(ParameterConverter::getInt(it->second));
                break;
                
            case 'D':
                _trainer->setMaxDepth(ParameterConverter::getInt(it->second));
                break;
                
            case 'W':
                _trainer->setMaxWidth(ParameterConverter::getInt(it->second));
                break;
                
            case 'B':
                _trainer->setUseBagging(ParameterConverter::getBool(it->second));
                break;
                
            case 'I':
                _trainer->setMaxLevelIterations(ParameterConverter::getInt(it->second));
                break;
                
            case 'P':
                _trainer->setSortParentNodes(ParameterConverter::getBool(it->second));
                break;
                
            case 'd':
                dumpSettings = ParameterConverter::getBool(it->second);
                break;
                
            case 'V':
                validationSetFileName = it->second;
                break;
               
            case 'v':
                validationLevel = ParameterConverter::getInt(it->second);
                break;
        }
        
        validationLevel = std::max(std::abs(validationLevel), 0);
        validationLevel = std::min(validationLevel, 3);
        
        // If there is a validation set, the validation level should be at least 1
        if (validationSetFileName.size() > 0)
        {
            validationLevel = std::max(std::abs(validationLevel), 1);
        }
        _trainer->setValidationLevel(validationLevel);
    }
}

int TrainCLIFunction::execute()
{
    dumpSettings = false;
    
    // There must be a model file and a training set
    if (getArguments()->getArguments().size() != 2)
    {
        std::cout << "Please use the command as follows:" << std::endl;
        std::cout << " $ jungle train [parameters] {trainingset} {model}" << std::endl;
        std::cout << "See '$ jugle help train' for more information." << std::endl;
        return 1;
    }
    
    
    // Create a trainer and load the given parameters
    JungleTrainer::ptr jungleTrainer = JungleTrainer::Factory::create();
    // In CLI we always work in verbose mode
    jungleTrainer->setVerboseMode(true);
    loadParametersToTrainer(jungleTrainer);
    
    if (dumpSettings)
    {
        std::cout << "Settings dump:" << std::endl;
        std::cout << "numFeatureSamples " << jungleTrainer->getNumFeatureSamples() << std::endl;
        std::cout << "maxDepth " << jungleTrainer->getMaxDepth() << std::endl;
        std::cout << "maxWidth " << jungleTrainer->getMaxWidth() << std::endl;
        std::cout << "useBagging " << jungleTrainer->getUseBagging() << std::endl;
        std::cout << "maxLevelIterations " << jungleTrainer->getMaxLevelIterations() << std::endl;
        std::cout << "numDAGs " << jungleTrainer->getNumDAGs() << std::endl;
        std::cout << "numTrainingSamples " << jungleTrainer->getNumTrainingSamples() << std::endl;
        std::cout << "sortParentNodes " << jungleTrainer->getSortParentNodes() << std::endl << std::endl;
    }
    
    // Load the training set
    std::cout << "Loading training set" << std::endl;
    TrainingSet::ptr trainingSet = TrainingSet::Factory::createFromFile(getArguments()->getArguments().at(0), false);
    TrainingSet::ptr testSet;
    
    // If there is a validation set, load it
    if (validationLevel > 0 && validationSetFileName != "")
    {
        std::cout << "Loading validation set" << std::endl;
        testSet = TrainingSet::Factory::createFromFile(validationSetFileName, false);
        jungleTrainer->setValidationSet(testSet);
    }
    
    std::cout << std::endl;
    
    // Train the jungle
    // start timing
    boost::timer t; 
    Jungle::ptr jungle = jungleTrainer->train(trainingSet);
    std::cout << "Training time: " << static_cast<float>(t.elapsed()) << "s\n";
    
    std::cout << std::endl;
  
    // Display some error statistics
    TrainingStatistics::ptr statisticsTool = TrainingStatistics::Factory::create();
    
    std::cout << "Training error: " << statisticsTool->trainingError(jungle, trainingSet) << std::endl;
    
    if (validationLevel > 0 && testSet)
    {
        std::cout << "Test error: " << statisticsTool->trainingError(jungle, testSet) << std::endl;
        TrainingSet::freeTrainingExamples(testSet);
    }
    TrainingSet::freeTrainingExamples(trainingSet);
    
    // Save the jungle in a file
    Jungle::Factory::serialize(jungle, getArguments()->getArguments().at(1));
    
    delete jungleTrainer;
    
    return 0;
}


const char* TrainCLIFunction::help()
{
    return  "USAGE \n"
            " $ jungle train [parameters] {trainingset} {model} \n\n"
            "PARAMETERS\n"
            " {trainingset} The filename of a training set to train on\n"
            " {model}       The output filename of the model file\n"
            " -M [int]      The number of DAGs that are trained\n"
            " -N [int]      The number of training examples that are sampled per DAG\n"
            " -F [int]      The number of features that are sampled per node\n"
            " -D [int]      Maximum depth of each DAG\n"
            " -W [int]      Maximum width of each DAG\n"
            " -B [bool]     Whether of not to use bagging. See also -N\n"
            " -I [int]      Maximum number of iterations at each level\n"
            " -P [bool]     Whether or not the parent nodes shall be sorted by their entropy\n"
            " -V [string]   The filename of a validation set\n"
            " -v [int]      Validation level. 1: After training, 2: After each DAG, 3: After each level \n\n"
            "DESCRIPTION\n"
            " This command trains a new decision jungle on the training set\n"
            " stored in {trainingset}. The trained model will be saved in\n"
            " {model}.\n";
}


const char* TrainCLIFunction::shortHelp()
{
    return "Trains a new decision jungle on a training set";
}

int VersionCLIFunction::execute()
{
    std::cout << "Everything except sse.h and fastlog.h is licensed under the following BSD license:" << std::endl;
    std::cout << std::endl;
    std::cout << "Copyright (c) 2014, Tobias Pohlen <tobias.pohlen@rwth-aachen.de>" << std::endl;
    std::cout << "All rights reserved." << std::endl;
    std::cout << std::endl;
    std::cout << "Redistribution and use in source and binary forms, with or without" << std::endl;
    std::cout << "modification, are permitted provided that the following conditions are met:" << std::endl;
    std::cout << "    * Redistributions of source code must retain the above copyright" << std::endl;
    std::cout << "      notice, this list of conditions and the following disclaimer." << std::endl;
    std::cout << "    * Redistributions in binary form must reproduce the above copyright" << std::endl;
    std::cout << "      notice, this list of conditions and the following disclaimer in the" << std::endl;
    std::cout << "      documentation and/or other materials provided with the distribution." << std::endl;
    std::cout << "    * The names of its contributors may not be used to endorse or promote products" << std::endl;
    std::cout << "      derived from this software without specific prior written permission." << std::endl;
    std::cout << std::endl;
    std::cout << "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND" << std::endl;
    std::cout << "ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED" << std::endl;
    std::cout << "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE" << std::endl;
    std::cout << "DISCLAIMED. IN NO EVENT SHALL TOBIAS POHLEN BE LIABLE FOR ANY" << std::endl;
    std::cout << "DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES" << std::endl;
    std::cout << "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;" << std::endl;
    std::cout << "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND" << std::endl;
    std::cout << "ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT" << std::endl;
    std::cout << "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS" << std::endl;
    std::cout << "SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE." << std::endl;
        
    std::cout << std::endl;
    std::cout << "Fast log2 approximation (sse.h, fastlog.h) Copyright (C) 2011 Paul Mineiro." << std::endl;
    std::cout << "Further information under: https://code.google.com/p/fastapprox/" << std::endl;
    return 0;
}


const char* VersionCLIFunction::help()
{
    return  "USAGE \n"
            " $ jungle version \n\n"
            "PARAMETERS\n"
            " There are no parameters for this command\n\n"
            "DESCRIPTION\n"
            " This command displays the version and license information\n"
            " for this build.\n";
}

const char* VersionCLIFunction::shortHelp()
{
    return "Displays the version information and license information for this build";
}