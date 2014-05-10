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
#include <random>


using namespace decision_jungle;

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
    catch (CLIFunctionNotFoundException & e) {
        std::cout << "There was an error." << std::endl;
        std::cout << "Message: " << e.what() << std::endl;
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
        if (argv[i][1] == '-')
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
                result->getParameters()[parameter.substr(1, equalPos - 1)] = parameter.substr(equalPos, parameter.size() - equalPos);
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
    ClassMap::iterator it = functionMap.find(functionName);
    if (it == functionMap.end())
    {
        // The function name is unknown
        throw CLIFunctionNotFoundException("Unknown function name.");
    }
    
    AbstractCLIFunction::ptr result(it->second());
    // Assign the argument bag
    result->arguments = _argumentBag;
    
    return result;
}

int HelpCLIFunction::execute()
{
    std::cout << "Execute help" << std::endl;
    return 0;
}


const char* HelpCLIFunction::help()
{
    return "Long help";
}


const char* HelpCLIFunction::shortHelp()
{
    return "Short help";
}


int ClassifyCLIFunction::execute()
{
    std::cout << "Execute classify" << std::endl;
    return 0;
}


const char* ClassifyCLIFunction::help()
{
    return "Long help";
}


const char* ClassifyCLIFunction::shortHelp()
{
    return "Short help";
}


int TrainCLIFunction::execute()
{
    int N = 20000;
    
    TrainingSet::ptr trainingSet = TrainingSet::Factory::create();
    
    std::normal_distribution<> norm1(-1, 0.5);
    std::normal_distribution<> norm2(1, 0.5);
    std::normal_distribution<> norm3(25, 0.5);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::cout << "Create random training set" << std::endl;
    
    for (int i = 0; i < N/4; i++)
    {
        TrainingExample::ptr current = TrainingExample::Factory::createZeroInitialized(2, 0);
        (*current->getDataPoint())[0] = norm1(gen);
        (*current->getDataPoint())[1] = norm3(gen);
        trainingSet->push_back(current);
    }
    
    for (int i = N/4; i < N/2; i++)
    {
        TrainingExample::ptr current = TrainingExample::Factory::createZeroInitialized(2, 1);
        (*current->getDataPoint())[0] = norm2(gen);
        (*current->getDataPoint())[1] = norm2(gen);
        trainingSet->push_back(current);
    }
    
    for (int i = N/2; i < 3*N/4; i++)
    {
        TrainingExample::ptr current = TrainingExample::Factory::createZeroInitialized(2, 2);
        (*current->getDataPoint())[0] = norm1(gen);
        (*current->getDataPoint())[1] = norm2(gen);
        trainingSet->push_back(current);
    }
    
    for (int i = 3*N/4; i < N; i++)
    {
        TrainingExample::ptr current = TrainingExample::Factory::createZeroInitialized(2, 3);
        (*current->getDataPoint())[0] = norm2(gen);
        (*current->getDataPoint())[1] = norm1(gen);
        trainingSet->push_back(current);
    }
    
    JungleTrainer::ptr jungleTrainer = JungleTrainer::Factory::create();
    jungleTrainer->setNumDAGs(150);
    Jungle::ptr jungle = jungleTrainer->train(trainingSet);
    return 0;
}


const char* TrainCLIFunction::help()
{
    return "Long help";
}


const char* TrainCLIFunction::shortHelp()
{
    return "Short help";
}


int VersionCLIFunction::execute()
{
    std::cout << "Version: " << VERSION_MAJOR << "." << VERSION_MINOR << std::endl;
    std::cout << "Copyright (c) 2014 Tobias Pohlen <tobias.pohlen@rwth-aachen.de>." << std::endl;
    std::cout << "All rights reserved." << std::endl;
    std::cout << "Released under the GPL licence." << std::endl;
    return 0;
}


const char* VersionCLIFunction::help()
{
    return "Long help";
}


const char* VersionCLIFunction::shortHelp()
{
    return "Short help";
}
