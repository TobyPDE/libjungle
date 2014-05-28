#include <map>
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>


#include "jungle.h"
#include "jungleTrain.h"

using namespace decision_jungle;

PredictionResult::ptr DAGNode::predict(DataPoint::ptr featureVector) const
{
    // Does this node have child nodes?
    if (!left)
    {
        // Nope, return the class label
        
        // Compute the relative confidence
        if (classHistogram->getMass() > 0)
        {
            return PredictionResult::Factory::create(getClassLabel(), getClassHistogram()->at(getClassLabel())/classHistogram->getMass());
        }
        else
        {
            return PredictionResult::Factory::create(getClassLabel(), 0);
        }
    }
    else
    {
        if (featureVector->at(featureID) <= threshold)
        {
            return left->predict(featureVector);
        }
        else
        {
            return right->predict(featureVector);
        }
    }
}

PredictionResult::ptr Jungle::predict(DataPoint::ptr featureVector) const
{
    // Use a majority vote
    std::map<ClassLabel, double> votes;
    PredictionResult::ptr prediction;
    
    for (std::set<DAGNode::ptr>::iterator it = dags.begin(); it != dags.end(); ++it)
    {
        prediction = (*it)->predict(featureVector);
        // Did we already encounter this class?
        if (votes.find(prediction->getClassLabel()) != votes.end())
        {
            votes[prediction->getClassLabel()] = votes[prediction->getClassLabel()] + prediction->getConfidence();
        }
        else
        {
            votes[prediction->getClassLabel()] = prediction->getConfidence();
        }
    }
    
    // Find the best class
    double bestScore = 0;
    ClassLabel bestLabel = -1;
    
    for (std::map<ClassLabel, double>::iterator it = votes.begin(); it != votes.end(); ++it)
    {
        if (it->second > bestScore)
        {
            bestScore = it->second;
            bestLabel = it->first;
        }
    }
    
    return PredictionResult::Factory::create(bestLabel);
}

DataPoint::ptr DataPoint::Factory::createFromFileRow(const std::vector<std::string> & _row)
{
    DataPoint::ptr dataPoint = DataPoint::Factory::createZeroInitialized(_row.size());
    
    // Parse the string elements of the vector
    int counter = 0;
    for (std::vector<std::string>::const_iterator it = _row.begin(); it != _row.end(); ++it)
    {
        (*dataPoint)[counter++] = atof(it->c_str());
    }
    
    return dataPoint;
}


DataSet::ptr DataSet::Factory::createFromFile(const std::string & _fileName, bool _verboseMode)
{
    // Create a blank training set and load the file line by line
    DataSet::ptr trainingSet = DataSet::Factory::create();
    
    std::string data(_fileName);

    std::ifstream in(data.c_str());
    if (!in.is_open())
    {
        throw RuntimeException("Could not open data set file.");
    }

    // Count the number of lines in order to display the progress bar
    std::ifstream countFile(_fileName); 
    int lineCount = std::count(std::istreambuf_iterator<char>(countFile), std::istreambuf_iterator<char>(), '\n');
    
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(lineCount);
    
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    std::vector< std::string > row;
    std::string line;

    if (_verboseMode)
    {
        std::cout << "Loading data set from " << _fileName << std::endl;
    }
    while (std::getline(in,line))
    {
        if (_verboseMode)
        {
            progressBar->update();
        }
        
        Tokenizer tok(line);
        row.assign(tok.begin(),tok.end());

        // Do not consider blank line
        if (row.size() == 0) continue;
        
        // Create the corresponding data point by considering only the last columns
        DataPoint::ptr point = DataPoint::Factory::createFromFileRow(row);
        trainingSet->push_back(point);
    }
    std::cout << "Data set loaded. Number of examples: " << trainingSet->size() << std::endl ;
    
    return trainingSet;
}