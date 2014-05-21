#include "fastTrain.h"

using namespace decision_jungle;

float flog2( float x )
{
    union { float f; uint32_t i; } vx = { x };
    union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
    float y = float(vx.i); y *= 1.1920928955078125e-7f;
    return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

void FastDAG::train(FastTrainingSet* _trainingSet)
{
    // Get the total number of class labels
    _classCount = 0;
    for (FastTrainingSet::iterator it = _trainingSet->begin(); it != _trainingSet->end(); ++it)
    {
        if ((*it)->label > _classCount - 1)
        {
            _classCount = (*it)->label + 1;
        }
    }
    _trainingSetSize = _trainingSet->size();
    
    int maxParentNodeCount = this->getMaxWidth();
    int maxNodeCount = maxParentNodeCount * this->getMaxDepth();
    
    // Initialize all arrays that we need
    _parents = new FastNode[maxParentNodeCount];
    std::fill_n(_parents, sizeof(FastNode), maxParentNodeCount, 0);
    
    _children = new FastNode[maxParentNodeCount];
    std::fill_n(_children, sizeof(FastNode), maxParentNodeCount, 0);
    
    _virtualHistograms = new float[_classCount * 2 * maxParentNodeCount];
    std::fill_n(_virtualHistograms, sizeof(float), _classCount * 2 * maxParentNodeCount, 0);
    
    _childHistograms = new float[2];
    std::fill_n(_childHistograms, sizeof(float), 2, 0);
    
    _virtualEntropies = new float[2 * maxParentNodeCount];
    std::fill_n(_virtualEntropies, sizeof(float), 2 * maxParentNodeCount, 0);
    
    _childEntropies = new float[maxParentNodeCount];
    std::fill_n(_childEntropies, sizeof(float),  maxParentNodeCount, 0);
    
    leftChildren = new FastNode[maxNodeCount];
    std::fill_n(leftChildren, sizeof(FastNode), maxNodeCount, 0);
    
    rightChildren = new FastNode[maxNodeCount];
    std::fill_n(rightChildren, sizeof(FastNode), maxNodeCount, 0);
    
    featureIDs = new int[maxNodeCount];
    std::fill_n(featureIDs, sizeof(int), maxNodeCount, 0);
    
    thresholds = new double[maxNodeCount];
    std::fill_n(thresholds, sizeof(double), maxNodeCount, 0);
    
    labels = new ClassLabel[maxNodeCount];
    std::fill_n(labels, sizeof(ClassLabel), maxNodeCount, 0);
    
    // Initialize the arrays
    _parentTrainingSets = new FastTrainingSet[this->getMaxWidth()];
    
    // Current number of parent nodes
    int parentCount = 0;
    // Current number of child nodes
    int childCount = 0;
    
    // Set up the training environment
    FastNode root = nodeCount++;
    _parents[0] = root;
    
    parentCount = 1;
    childCount = 2;
    
    _parentTrainingSets[0]->insert(_parentTrainingSets[0]->end(), _trainingSet->begin(), _trainingSet->end());
    
    for (int level = 1; level <= this->getMaxDepth(); level++)
    {
        int width = std::min(this->getMaxWidth(), parentCount*2);
        
        std::fill_n(_virtualHistograms, sizeof(float), _classCount * 2 * maxParentNodeCount, 0);
        std::fill_n(_childHistograms, sizeof(float), _classCount * maxParentNodeCount, 0);
        std::fill_n(_virtualEntropies, sizeof(float), 2 * maxParentNodeCount, 0);
        std::fill_n(_childEntropies, sizeof(float),  maxParentNodeCount, 0);
    
        // Find a child node assignment
        int c = 1;
        for (int p = 0; p < parentCount; p++)
        {
            leftChildren[_parents[p]] = (c++ % childCount); 
            rightChildren[_parents[p]] = (c++ % childCount); 
            
            // Create a histogram over the parent training sets
            for (FastTrainingSet::iterator it = _parentTrainingSets[p]->begin(); it != _parentTrainingSets[p]->end(); ++it)
            {
                if ((*it)->featureVector[featureIDs[_parents[p]]] < thresholds[_parents[p]])
                {
                    *(_virtualEntropies + p*_classCount + (*it)->label)++;
                }
            }
            // normalize the histogram and compute the entropy
            for (int i = 0; i < _classCount; i++) 
            {
                *(_virtualEntropies + p*_classCount + i) /= _parentTrainingSets[p]->size();
                _virtualEntropies[p] += *(_virtualEntropies + p*_classCount + i) * flog2(*(_virtualEntropies + p*_classCount + i));
            }
        }
        
        // Find the optimal threshold and child assignment
        bool change = false;
        do {
            // Find the threshold
            for (int p = 0; p < parentCount; p++)
            {
                // Initialize the child histograms
                std::fill_n(_childHistograms, sizeof(float), 2, 0);
                
                for (int _p = 0; _p < parentCount; _p++)
                {
                    if (p == _p) continue;
                    
                    if (leftChildren[_parents[_p]] == leftChildren[_parents[p]])
                    {
                        int offset_from = 2 * _p * _classCount;
                        
                        // Copy the histograms
                        for (int c = 0; c < _classCount; c++)
                        {
                            *(_childHistograms + c) += *(_virtualHistograms + offset_from + c);
                        }
                    }
                    
                    if (leftChildren[_parents[_p]] == rightChildren[_parents[p]])
                    {
                        int offset_from = 2 * _p * _classCount;
                        
                        // Copy the histograms
                        for (int c = 0; c < _classCount; c++)
                        {
                            *(_childHistograms + c + _classCount) += *(_virtualHistograms + offset_from + c);
                        }
                    }
                    
                    if (rightChildren[_parents[_p]] == leftChildren[_parents[p]])
                    {
                        int offset_from = (2*_p + 1) * _classCount;
                        
                        // Copy the histograms
                        for (int c = 0; c < _classCount; c++)
                        {
                            *(_childHistograms + c) += *(_virtualHistograms + offset_from + c);
                        }
                    }
                    
                    if (rightChildren[_parents[_p]] == rightChildren[_parents[p]])
                    {
                        int offset_from = (2*_p + 1) * _classCount;
                        
                        // Copy the histograms
                        for (int c = 0; c < _classCount; c++)
                        {
                            *(_childHistograms + c + _classCount) += *(_virtualHistograms + offset_from + c);
                        }
                    }
                }
                
                // FIXME
                for (int f = 0; f < _featureDim; f++)
                {
                    // Sort the training set according to the current feature dimension
                    TrainingExampleComparator compare(f);
                    std::sort(_parentTrainingSets[p]->begin(), _parentTrainingSets[p]->end(), compare);
                    
                    for (int _c = 0; _c < childCount; _c++)
                    {
                        
                    }
                    
                }
            }
        } while (change);
    }
}
