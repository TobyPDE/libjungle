#include <map>

#include "jungle.h"

using namespace decision_jungle;

PredictionResult::ptr DAGNode::predict(DataPoint::ptr featureVector) const
{
    // Does this node have child nodes?
    if (!left)
    {
        // Nope, return the class label
        return PredictionResult::Factory::create(getClassLabel(), getClassHistogram()->at(getClassLabel()));
    }
    else
    {
        if (featureVector->at(featureID) < threshold)
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
    std::map<ClassLabel, float> votes;
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
    float bestScore = 0;
    ClassLabel bestLabel = -1;
    
    for (std::map<ClassLabel, float>::iterator it = votes.begin(); it != votes.end(); ++it)
    {
        if (it->second > bestScore)
        {
            bestScore = it->second;
            bestLabel = it->first;
        }
    }
    
    return PredictionResult::Factory::create(bestLabel);
}

ClassHistogram::ptr Statistics::predictionHistogram(Jungle::ptr _jungle, DataSet::ptr _dataSet)
{
    ClassHistogram::ptr histogram = ClassHistogram::Factory::createEmpty(0);
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(_dataSet->size());
    
    for (DataSet::iterator it = _dataSet->begin(); it != _dataSet->end(); ++it)
    {
        // Display the progress bar
        if (this->getVerboseMode())
        {
            progressBar->update();
        }
        
        // Predict the class label
        ClassLabel label = _jungle->predict(*it)->getClassLabel();
        
        // Did we encounter the class label before?
        if (histogram->size() < label)
        {
            // Yes
            (*histogram)[label] = (*histogram)[label] + 1;
        }
        else
        {
            // No
            (*histogram)[label] = 1;
        }
    }
    
    return histogram;
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
