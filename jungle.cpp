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
