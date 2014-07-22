
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include <cstdlib>
#include <boost/tokenizer.hpp>
#include "jungleTrain.h"
#include "config.h"
#if DecisionJungle_USE_OPEN_MPI
    #include "omp.h"
#endif

using namespace LibJungle;

void AbstractTrainer::validateParameters() throw(ConfigurationException)
{
    if (maxDepth < 1)
    {
        throw ConfigurationException("max depth must be greater than 0.");
    }
    if (maxWidth < 0)
    {
        throw ConfigurationException("max width must be greater than 0.");
    }
}

void DAGTrainer::validateParameters() throw(ConfigurationException)
{
    AbstractTrainer::validateParameters();
    
    if (trainingSet->size() < 1)
    {
        throw ConfigurationException("There must be at least one training example.");
    }
    
    // Check if all training examples have the same feature dimension
    featureDimension = (*trainingSet->begin())->getDataPoint()->size();
    classCount = 0;
    
    const size_t trainingSetSize = trainingSet->size();
    for (size_t i = 0; i < trainingSetSize; i++)
    {
        TrainingExample* current = (*trainingSet)[i];
        
        if (static_cast<int>(current->getDataPoint()->size()) != featureDimension)
        {
            throw ConfigurationException("All data points must have the same feature dimension.");
        }
        // Check if all class labels are > 0
        if (current->getClassLabel() < 0)
        {
            throw ConfigurationException("All class labels must be greater than or equal to 0.");
        }
        // Update the class counter
        if (classCount < current->getClassLabel())
        {
            classCount = current->getClassLabel();
        }
    }
    classCount++;
    
    // Sample the features for this DAG
    if (getNumFeatureSamples() == -1)
    {
        // Automatically select the number 
        setNumFeatureSamples(static_cast<int>(std::floor(std::sqrt(featureDimension))));
    }
    
    // Check if the number of features to sample is valid
    if (getNumFeatureSamples() < 1 || getNumFeatureSamples() > featureDimension)
    {
        throw ConfigurationException("The number of features must be in [1, featureDimension].");
    }
}

    
// Sample the features
void DAGTrainer::getSampledFeatures(std::vector<int> & sampledFeature)
{
    std::uniform_int_distribution<int> dist(0, featureDimension - 1);
    std::random_device rd;
    std::default_random_engine gen(rd());
    
    for (int i = 0; i < getNumFeatureSamples(); i++)
    {
        sampledFeature.push_back(dist(gen));
    }
}

TrainingDAGNode::ptr DAGTrainer::train() throw(ConfigurationException, RuntimeException)
{
    // Only train the DAG if all parameters are valid
    validateParameters();
    
    // Start growing the DAG
    NodeRow parentNodes;
    
    // Create the root node
    // FIXME
    TrainingDAGNode::ptr root = TrainingDAGNode::Factory::create(this);
    // The root node gets all the training data
    root->getTrainingSet()->insert(root->getTrainingSet()->end(), trainingSet->begin(), trainingSet->end());
    // Set the class histogram and class label for the node
    root->updateHistogramAndLabel();
    // Add the root node to the initial parent level of training
    parentNodes.push_back(root);
    
    // The number of child nodes on the next level
    int childNodeCount = 0;
    
    TrainingStatistics::ptr statisticsTool = TrainingStatistics::Factory::create();
    Jungle::ptr jungle = Jungle::Factory::create();
    jungle->getDAGs().insert(root);
    
    for (int level = 1; level <= getMaxDepth(); level++)
    {
        // Determine the number of child nodes
        childNodeCount = std::min(static_cast<int>(parentNodes.size()) * 2, getMaxWidth());
        
        // Train the level
        parentNodes = trainLevel(parentNodes, childNodeCount);
        
        if (getVerboseMode())
        {
            if (getValidationLevel() >= 3)
            {
                if (getValidationSet())
                {
                    printf("level: %5d, nodes: %6d, training error: %1.6f, test error: %1.6f \n", level, static_cast<int>(parentNodes.size()), statisticsTool->trainingError(jungle, trainingSet), statisticsTool->trainingError(jungle, getValidationSet()));
                }
                else
                {
                    printf("level: %5d, nodes: %6d, training error: %1.6f\n", level, static_cast<int>(parentNodes.size()), statisticsTool->trainingError(jungle, trainingSet));
                }
                std::cout.flush();
            }
        }
        
        std::cout.flush();
    
        // Stop when there is nothing more to do
        if (parentNodes.size() == 0)
        {
            break;
        }
    }
    jungle->getDAGs().erase(jungle->getDAGs().begin());
    
    return root;
}

NodeRow DAGTrainer::trainLevel(NodeRow &parentNodes, int childNodeCount)
{
    // Sort the parent nodes decreasing by their entropy
    if (getSortParentNodes())
    {
        NodeEntropyComparator compare;
        std::sort(parentNodes.begin(), parentNodes.end(), compare);
    }
    
    // Initialize the parent level
    // We need a counter in order to assign the parent to some virtual children
    int vChildren = 0;
    const size_t parentNodeSize = parentNodes.size();
    for (size_t i = 0; i < parentNodeSize; i++)
    {
        TrainingDAGNode* current = parentNodes[parentNodeSize - i - 1];
        current->setThreshold(0);
        current->setFeatureID(0);
        current->updateLeftRightHistogram();
        
        // Assign the child nodes
        if (current->isPure())
        {
            current->setTempLeft(vChildren % childNodeCount);
            current->setTempRight(vChildren++ % childNodeCount);
        }
        else
        {
            current->setTempLeft(vChildren++ % childNodeCount);
            current->setTempRight(vChildren++ % childNodeCount);
        }
    }
    
    // Adjust the thresholds and child assignments until nothing changes anymore
    bool change = false;
    int iterationCounter = 0;
    bool isTreeLevel = (static_cast<int>(parentNodes.size()) * 2 == childNodeCount);
    do
    {
        change = false;
        for (size_t i = 0; i < parentNodeSize; i++)
        {
            TrainingDAGNode* current = parentNodes[i];
            // Pure nodes don't need a threshold
            if (current->isPure()) continue;
            
            // Find the new optimal threshold
            if (current->findThreshold(parentNodes))
            {
                change = true;
            }
        }
        
        // If this is a tree (e.g. 2 * #parent = #children), we don't need to train the child node assignments and are 
        // done at this point
        if (isTreeLevel) break;
        
        for (size_t i = 0; i < parentNodeSize; i++)
        {
            TrainingDAGNode* current = parentNodes[i];
            // Pure nodes must have left=right
            if (current->isPure())
            {
                // Find the new optimal child assignment for both pointers
                if (current->findCoherentChildNodeAssignment(parentNodes, childNodeCount))
                {
                    change = true;
                }
            }
            else
            {
                // Find the new optimal child assignment for the left pointer
                if (current->findRightChildNodeAssignment(parentNodes, childNodeCount))
                {
                    change = true;
                }
                // Find the new optimal child assignment for the right pointer
                if (current->findLeftChildNodeAssignment(parentNodes, childNodeCount))
                {
                    change = true;
                }
            }
        }
    }
    while (change && ++iterationCounter < getMaxLevelIterations());

    // Determine whether or not the row training shall be performed
    // Get the entropy of the parent row in order to determine whether or not the split shall be performed
    RowEntropyErrorFunction parentErrorFunction(parentNodes);
    ChildRowEntropyErrorFunction childErrorFunction(parentNodes, childNodeCount);
    float parentEntropy = parentErrorFunction.error();
    float childEntroy = childErrorFunction.error();
    // Do not perform the split if it increases the energy
    if (std::abs(parentEntropy) - std::abs(childEntroy) <= 1e-6)
    {
        return NodeRow();
    }
    
    // Create the child nodes
    NodeRow childNodes(childNodeCount);
    // Memorize which child nodes don't have a parent node in this variable
    bool* noParentNode = new bool[childNodeCount];
    
    for (int i = 0; i < childNodeCount; i++)
    {
        childNodes[i] = TrainingDAGNode::Factory::create(this);
        noParentNode[i] = true;
    }

    // Assign the parent to their child nodes and set the training sets
    for (size_t i = 0; i < parentNodeSize; i++)
    {
        TrainingDAGNode* current = parentNodes[i];
        int leftNode = current->getTempLeft();
        int rightNode = current->getTempRight();
        
        // Assign the parent to the children
        current->setLeft(childNodes[leftNode]);
        current->setRight(childNodes[rightNode]);

        // Propagate the training set
        TrainingSet::ptr parentTrainingSet = current->getTrainingSet();
        const size_t parentTrainingSetSize = parentTrainingSet->size();
        
        for (size_t j = 0; j < parentTrainingSetSize; j++)
        {
            TrainingExample* currentTrainingExample = parentTrainingSet->at(j);
            // Determine whether or not this example belongs to the left or right child node
            const float featureValue = currentTrainingExample->getDataPoint()->at(current->getFeatureID());
            if (featureValue <= current->getThreshold())
            {
                // Left child node
                childNodes[leftNode]->getTrainingSet()->push_back(currentTrainingExample);
            }
            else
            {
                // Right child node
                childNodes[rightNode]->getTrainingSet()->push_back(currentTrainingExample);
            }
            noParentNode[leftNode] = false;
            noParentNode[rightNode] = false;
        }
    }
    
    for (size_t j = 0; j < parentNodeSize; j++)
    {
        parentNodes[j]->getTrainingSet()->clear();
    }

    // It might happen, that a threshold was selected such that a child node
    // did not receive any training examples. In this case, unify the child nodes
    for (size_t i = 0; i < parentNodeSize; i++)
    {
        TrainingDAGNode* current = parentNodes[i];
        int leftNode = current->getTempLeft();
        int rightNode = current->getTempRight();
        
        if (childNodes[leftNode]->getTrainingSet()->size() == 0)
        {
            current->setLeft(childNodes[rightNode]);
            current->setTempLeft(rightNode);
            noParentNode[leftNode] = true;
        }
        else if (childNodes[rightNode]->getTrainingSet()->size() == 0)
        {
            current->setRight(childNodes[leftNode]);
            current->setTempRight(leftNode);
            noParentNode[rightNode] = true;
        }
    }
    
    
    for (int i = 0; i < childNodeCount; i++)
    {
        // Select the class label and compute the class histogram for this child node
        childNodes[i]->updateHistogramAndLabel();
    }
    
    NodeRow returnChildNodes;
    
    
    // Decide which nodes to split further
    for (int i = 0; i < childNodeCount; i++)
    {
        // Delete the nodes without parents and split the other ones
        if (noParentNode[i])
        {
            delete childNodes[i];
        }
        else
        {
            // This is a pure node, don't split it
            returnChildNodes.push_back(childNodes[i]);
        }
    }
    delete[] noParentNode;
    return returnChildNodes;
}

TrainingDAGNode::ptr TrainingDAGNode::Factory::create(DAGTrainerPtr trainer)
{
    TrainingDAGNode::ptr node = new TrainingDAGNode(trainer);

    // Initialize the parent parameters
    DAGNode::Factory::init(node, trainer->getClassCount());

    node->trainingSet = TrainingSet::Factory::create();
    
    // Initialize the training parameters
    node->setTempLeft(0);
    node->setTempRight(0);
    node->pure = false;

    // Initialize the histograms
    node->getLeftHistogram()->resize(trainer->getClassCount());
    node->getRightHistogram()->resize(trainer->getClassCount());
    node->getClassHistogram()->resize(trainer->getClassCount());
    
    return node;
}

void TrainingDAGNode::resetLeftRightHistogram()
{
    // The left one becomes zero, the right one becomes the node histogram
    ClassHistogram* nodeHistogram = getClassHistogram();

    for (int i = 0; i < trainer->getClassCount(); i++)
    {
        leftHistogram.set(i, 0);
        rightHistogram.set(i, nodeHistogram->at(i));
    }
}


void TrainingDAGNode::updateHistogramAndLabel()
{
    // Compute the histogram
    TrainingUtil::computHistogram(*getClassHistogram(), getTrainingSet());
    // Get the best class label
    setClassLabel(TrainingUtil::histogramArgMax(*getClassHistogram()));
    
    pure = TrainingUtil::histogramIsDirichlet(*getClassHistogram());
    
    entropy = getClassHistogram()->entropy();
}

bool TrainingDAGNode::findThreshold(NodeRow & parentNodes)
{
    // If there are no training examples, there is nothing to train
    if (trainingSet->size() == 0) return false;
    
    ThresholdEntropyErrorFunction error(parentNodes, this); 
    
    error.initHistograms();
    // Compute the current error in order to find a better threshold
    float bestEntropy = error.error();
    
    error.resetHistograms();
            
    // We need to save the current settings in order to restore them after optimization because we modify the object
    // in order to evaluate the error function
    int bestFeatureID = getFeatureID();
    float bestThreshold = getThreshold();
    
    float currentEntropy = 0;
    
    // Return flag to notify the calling optimizer whether or not we changed the threshold
    bool changed = false;
    
    this->resetLeftRightHistogram();
    
    // Iterate over all sampled features
    std::vector<int> sampledFeatures;
    trainer->getSampledFeatures(sampledFeatures);
    const size_t sampledFeaturesSize = sampledFeatures.size();
    const size_t trainingSetSize = trainingSet->size();
    for (size_t i = 0; i < sampledFeaturesSize; i++)
    {
        const int feature = sampledFeatures[i];
        setFeatureID(feature);
        
        // Sort the training set according to the current feature dimension
        TrainingExampleComparator compare(getFeatureID());
        std::sort(trainingSet->begin(), trainingSet->end(), compare);
        
        // Initialize the virtual left/right histograms
        error.resetHistograms();
        
        // Test all possible splits
        for (size_t j = 0; j < trainingSetSize - 1; j++)
        {
            TrainingExample* it = trainingSet->at(j);
            TrainingExample* itp1 = trainingSet->at(j+1);
            // Choose the threshold as value between the two adjacent elements
            setThreshold( (it->getDataPoint()->at(feature) + itp1->getDataPoint()->at(feature) ) / 2 );
            
            // Update the histograms
            error.move(it->getClassLabel());
            
            // Get the current entropy
            currentEntropy = error.error();
            
            // Only accept the split if the entropy decreases and the threshold is not insignificant
            if (currentEntropy < bestEntropy && ( itp1->getDataPoint()->at(feature) - it->getDataPoint()->at(feature)) >= 1e-6)
            {
                // Select this feature and this threshold
                bestFeatureID = feature;
                bestThreshold = getThreshold();
                bestEntropy = currentEntropy;
                changed = true;
            }
        }
    }
    // Restore the arg min settings
    setFeatureID(bestFeatureID);
    setThreshold(bestThreshold);
    updateLeftRightHistogram();
    
    return changed;
}

bool TrainingDAGNode::findLeftChildNodeAssignment(NodeRow & parentNodes, int childNodeCount)
{
    // If there are no training examples, there is nothing to train
    if (trainingSet->size() == 0) return false;
    
    // Create the error function
    AssignmentEntropyErrorFunction error(parentNodes, this, childNodeCount);
    error.initHistograms();
    
    // Save the currently best settings
    int selectedLeft = getTempLeft();
    
    float bestEntropy = error.error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int cLeft = 0; cLeft < childNodeCount; cLeft++)
    {
        // Test this assignment
        setTempLeft(cLeft);

        // Get the error
        currentEntropy = error.error();

        // Is this better?
        if (currentEntropy < bestEntropy)
        {
            // it is
            selectedLeft = cLeft;
            bestEntropy = currentEntropy;
            changed = true;
        }
    }
    
    // Restore the arg min setting
    setTempLeft(selectedLeft);
    
    return changed;
}

bool TrainingDAGNode::findRightChildNodeAssignment(NodeRow & parentNodes, int childNodeCount)
{
    // If there are no training examples, there is nothing to train
    if (trainingSet->size() == 0) return false;
    
    // Create the error function
    AssignmentEntropyErrorFunction error(parentNodes, this, childNodeCount);
    error.initHistograms();

    // Save the currently best settings
    int selectedRight = getTempRight();
    
    float bestEntropy = error.error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int cRight = 0; cRight < childNodeCount; cRight++)
    {
        // Test this assignment
        setTempRight(cRight);

        // Get the error
        currentEntropy = error.error();

        // Is this better?
        if (currentEntropy < bestEntropy)
        {
            // it is
            selectedRight = cRight;
            bestEntropy = currentEntropy;
            changed = true;
        }
    }
    
    // Restore the arg min setting
    setTempRight(selectedRight);
    
    return changed;
}

bool TrainingDAGNode::findCoherentChildNodeAssignment(NodeRow & parentNodes, int childNodeCount)
{
    // If there are no training examples, there is nothing to train
    if (trainingSet->size() == 0) return false;
    
    // Create the error function
    AssignmentEntropyErrorFunction error(parentNodes, this, childNodeCount);
    error.initHistograms();
    
    // Save the currently best settings
    int selectedRight = getTempRight();
    int selectedLeft = getTempLeft();
    
    float bestEntropy = error.error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int current = 0; current < childNodeCount; current++)
    {
        // Test this assignment
        setTempRight(current);
        setTempLeft(current);
        
        // Get the error
        currentEntropy = error.error();

        // Is this better?
        if (currentEntropy < bestEntropy)
        {
            // it is
            selectedRight = current;
            selectedLeft = current;
            bestEntropy = currentEntropy;
            changed = true;
        }
    }
    
    // Restore the arg min setting
    setTempRight(selectedRight);
    setTempLeft(selectedLeft);
    
    return changed;
}

TrainingSet::ptr TrainingSet::Factory::createBySampling(TrainingSet::ptr _trainingSet, int n)
{
    // Create a distribution over the training set
    std::uniform_int_distribution<int> dist(0,_trainingSet->size() - 1);
    std::random_device rd;
    std::default_random_engine gen(rd());
    
    TrainingSet::ptr result = TrainingSet::Factory::create();
    
    for (int i = 0; i < n; i++)
    {
        result->push_back((*_trainingSet)[dist(gen)]);
    }
    
    return result;
}

TrainingExample::ptr TrainingExample::Factory::createFromFileRow(const std::vector<std::string> & _row)
{
    // There must be at least two entries. Otherwise the vector was empty or the class label
    // was missing
    if (_row.size() < 2)
    {
        throw RuntimeException("Illegal training set row.");
    }
    
    // Create the corresponding data point by considering only the last columns
    std::vector<std::string> dataPointRow;
    dataPointRow.insert(dataPointRow.begin(), _row.begin() + 1, _row.end());
    
    return TrainingExample::Factory::create(DataPoint::Factory::createFromFileRow(dataPointRow), atoi(_row[0].c_str()));
}

TrainingSet::ptr TrainingSet::Factory::createFromFile(const std::string & _fileName, bool _verboseMode)
{
    // Create a blank training set and load the file line by line
    TrainingSet::ptr trainingSet = TrainingSet::Factory::create();
    
    std::string data(_fileName);

    std::ifstream in(data.c_str());
    if (!in.is_open())
    {
        throw RuntimeException("Could not open training set file.");
    }

    // Count the number of lines in order to display the progress bar
    std::ifstream countFile(_fileName); 
    int lineCount = std::count(std::istreambuf_iterator<char>(countFile), std::istreambuf_iterator<char>(), '\n');
    countFile.close();
    
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(lineCount);
    
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    std::vector< std::string > row;
    std::string line;

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
        
        // Load the training example to the training set
        TrainingExample::ptr example = TrainingExample::Factory::createFromFileRow(row);
        trainingSet->push_back(example);
    }
    
    in.close();
    
    return trainingSet;
}

DAGTrainer::ptr DAGTrainer::Factory::createFromJungleTrainer(JungleTrainer::ptr _jungleTrainer, TrainingSet::ptr _trainingSet)
{
    DAGTrainer::ptr result = createForTraingSet(_trainingSet);
    
    // Transfer all parameter
    result->setMaxDepth(_jungleTrainer->getMaxDepth());
    result->setMaxWidth(_jungleTrainer->getMaxWidth());
    result->setVerboseMode(_jungleTrainer->getVerboseMode());
    result->setNumFeatureSamples(_jungleTrainer->getNumFeatureSamples());
    result->setUseBagging(_jungleTrainer->getUseBagging());
    result->setMaxLevelIterations(_jungleTrainer->getMaxLevelIterations());
    result->setValidationLevel(_jungleTrainer->getValidationLevel());
    result->setValidationSet(_jungleTrainer->getValidationSet());
    result->setSortParentNodes(_jungleTrainer->getSortParentNodes());
    
    return result;
}


void AbstractTrainer::Factory::init(AbstractTrainer::ptr _trainer)
{
    _trainer->maxDepth = 256;
    _trainer->maxWidth = 128;
    _trainer->verboseMode = false;
    _trainer->useBagging = false;
    _trainer->maxLevelIterations = 55;
    _trainer->validationLevel = 0;
    // -1 means that the number of features to sample will be determined automatically
    _trainer->numFeatureSamples = -1;
    _trainer->sortParentNodes = true;
}

void JungleTrainer::Factory::init(JungleTrainer::ptr _trainer)
{
    AbstractTrainer::Factory::init(_trainer);
    
    // -1 means that the number of features to sample will be determined automatically
    _trainer->numTrainingSamples = -1;
    _trainer->numDAGs = 1;
}

Jungle::ptr JungleTrainer::train(TrainingSet::ptr trainingSet) throw(ConfigurationException, RuntimeException)
{
    // If the number of training examples is set to -1, we determine the number automatically
    if (numTrainingSamples == -1)
    {
        numTrainingSamples = std::min(static_cast<int>(trainingSet->size()), static_cast<int>(std::floor(trainingSet->size() * 5 / static_cast<float>(numDAGs))));
    }
    
    Jungle::ptr jungle = Jungle::Factory::create();
    
    if (getVerboseMode())
    {
        printf("Start training\n");
        printf("Number of training examples: %d\n", static_cast<int>(trainingSet->size()));
        if (getUseBagging())
        {
            printf("Number of examples per DAG: %d\n", getNumTrainingSamples());
        }
        printf("Number of DAGs to train: %d\n", getNumDAGs());
    }
    
    // Display some error statistics
    TrainingStatistics::ptr statisticsTool = TrainingStatistics::Factory::create();

    #pragma omp parallel for
    for (int i = 0; i < numDAGs; i++)
    {
        #pragma omp critical
        {
            if (getVerboseMode())
            {
                std::cout << "Train DAG " << (i+1) << "/" << getNumDAGs() << std::endl;
            }
        }
        
        // Create a training set for each DAG by sampling from the given training set
        TrainingSet::ptr sampledSet = trainingSet;
        if (getUseBagging())
        {
            sampledSet = TrainingSet::Factory::createBySampling(trainingSet, numTrainingSamples);
        }
        
        DAGTrainer::ptr trainer = DAGTrainer::Factory::createFromJungleTrainer(this, sampledSet);\
        TrainingDAGNode::ptr dag = trainer->train();

        #pragma omp critical
        {
            jungle->getDAGs().insert(dag);
            if (getVerboseMode())
            {
                std::cout << "DAG completed\n";
                std::cout << "Training error: " << statisticsTool->trainingError(jungle, trainingSet) << std::endl;
                if (getValidationLevel() >= 2 && getValidationSet())
                {
                    std::cout << "Test error: " << statisticsTool->trainingError(jungle, getValidationSet()) << std::endl;
                }
                std::cout << "----------------------------\n";
            }
        }

        delete trainer;
    }
    
    return jungle;
}

float TrainingStatistics::trainingError(Jungle::ptr _jungle, TrainingSet::ptr _trainingSet)
{
    // Calculate the training error
    float error = 0;
    for (TrainingSet::iterator iter = _trainingSet->begin(); iter != _trainingSet->end(); ++iter)
    {
        if ((*iter)->getClassLabel() != _jungle->predict((*iter)->getDataPoint())->getClassLabel())
        {
            error++;
        }
    }
    
    // Calculate the relative error
    if (_trainingSet->size()  > 0)
    {
        error = error/static_cast<float>(_trainingSet->size());
    }
    
    return error;
}

void TrainingDAGNode::updateLeftRightHistogram()
{
    leftHistogram.reset();
    rightHistogram.reset();

    const size_t trainingSetSize = trainingSet->size();
    for(size_t i = 0; i < trainingSetSize; i++)
    {
        TrainingExample* current = trainingSet->at(i);
        
        // Determine whether or not this example belongs to the left or right child node
        if (current->getDataPoint()->at(getFeatureID()) <= getThreshold())
        {
            // Left child node
            leftHistogram.addOne(current->getClassLabel());
        }
        else
        {
            // Right child node
            rightHistogram.addOne(current->getClassLabel());
        }
    }
}
