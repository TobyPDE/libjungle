
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include "jungleTrain.h"
#include <boost/tokenizer.hpp>
#include <cstdlib>

using namespace decision_jungle;

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
    if (minSplitCount < 0)
    {
        throw ConfigurationException("min split count must be greater than 0.");
    }
    if (minChildSplitCount < 0)
    {
        throw ConfigurationException("min child split count must be greater than 0.");
    }
    if (trainingMethod != AbstractErrorFunction::error_entropy && trainingMethod != AbstractErrorFunction::error_gini)
    {
        throw ConfigurationException("Unknown training method.");
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
    
    for (TrainingSet::iterator iter = trainingSet->begin(); iter != trainingSet->end(); ++iter)
    {
        if ((*iter)->getDataPoint()->size() != featureDimension)
        {
            throw ConfigurationException("All data points must have the same feature dimension.");
        }
        // Check if all class labels are > 0
        if ((*iter)->getClassLabel() < 0)
        {
            throw ConfigurationException("All class labels must be greater than or equal to 0.");
        }
        // Update the class counter
        if (classCount < (*iter)->getClassLabel())
        {
            classCount = (*iter)->getClassLabel();
        }
    }
    classCount++;
    
    // Check if the feature weights are correct
    if (getFeatureWeights().size() != featureDimension)
    {
        // FIXME: Notice the user that we use equal feature weights
        
        // Weight all features equally
        float weight = 1/static_cast<float>(featureDimension);
        getFeatureWeights().resize(featureDimension);
        
        for (int i = 0; i < featureDimension; i++)
        {
            getFeatureWeights()[i] = weight;
        }
    }
    
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
    std::set<TrainingDAGNode::ptr> parentNodes;
    
    // Create the root node
    // FIXME
    TrainingDAGNode::ptr root = TrainingDAGNode::Factory::create(this);
    // The root node gets all the training data
    root->getTrainingSet()->insert(root->getTrainingSet()->end(), trainingSet->begin(), trainingSet->end());
    // Set the class histogram and class label for the node
    root->updateHistogramAndLabel();
    // Add the root node to the initial parent level of training
    parentNodes.insert(root);
    
    // The number of child nodes on the next level
    int childNodeCount = 0;
    
    for (int level = 1; level <= getMaxDepth(); level++)
    {
        // Determine the number of child nodes
        childNodeCount = std::min(static_cast<int>(parentNodes.size()) * 2, getMaxWidth());
        
        printf("l: %d, #p: %d\n", level, static_cast<int>(parentNodes.size()));
        
        // Train the level
        parentNodes = trainLevel(parentNodes, childNodeCount);
        
        // Stop growing if there are no more child nodes to be split
        if (parentNodes.size() == 0)
        {
            break;
        }
    }
    
    return root;
}

std::set<TrainingDAGNode::ptr> DAGTrainer::trainLevel(std::set<TrainingDAGNode::ptr> &parentNodes, int childNodeCount)
{
    // Initialize the parent level
    // We need a counter in order to assign the parent to some virtual children
    int vChildren = 0;
    for (std::set<TrainingDAGNode::ptr>::iterator iter = parentNodes.begin(); iter != parentNodes.end(); ++iter)
    {
        // Initialize the thresholds for the parent nodes as they each had dedicated child nodes
        // FIXME
        AbstractErrorFunction::ptr nodeErrorFunction = AbstractErrorFunction::Factory::createNodeErrorFunction(getTrainingMethod(), *iter);
        (*iter)->resetLeftRightHistogram();
        (*iter)->setThreshold(0);
        (*iter)->findThreshold(nodeErrorFunction);
        
        // Assign the child nodes
        (*iter)->setTempLeft(vChildren++ % childNodeCount);
        (*iter)->setTempRight(vChildren++ % childNodeCount);
    }
    
    // Create the error function that measures the entropy on the child level of the DAG
    AbstractErrorFunction::ptr childErrorFunction = AbstractErrorFunction::Factory::createChildNodeRowErrorFunction(getTrainingMethod(), parentNodes, childNodeCount);
    // FIXME
    
    // Adjust the thresholds and child assignments until nothing changes anymore
    if (2*parentNodes.size() != childNodeCount) {
        bool change = false;
        do
        {
            change = false;
            for (std::set<TrainingDAGNode::ptr>::iterator iter = parentNodes.begin(); iter != parentNodes.end(); ++iter)
            {
                NewChildRowEntropyErrorFunction* __childErrorFunction = new NewChildRowEntropyErrorFunction(&parentNodes, *iter);
                __childErrorFunction->initHistograms();

                // Find the new optimal threshold
                if ((*iter)->findThreshold(__childErrorFunction))
                {
                    change = true;
                }
                delete __childErrorFunction;
                // Find the new optimal child assignment
                if ((*iter)->findRightChildNodeAssignment(childErrorFunction, childNodeCount))
                {
                    change = true;
                }
                // Find the new optimal child assignment
                if ((*iter)->findLeftChildNodeAssignment(childErrorFunction, childNodeCount))
                {
                    change = true;
                }
            }
        }
        while (change);
    }

    // Determine whether or not the row training shall be performed
    // Get the entropy of the parent row in order to determine whether or not the split shall be performed
    AbstractErrorFunction::ptr parentErrorFunction = AbstractErrorFunction::Factory::createNodeRowErrorFunction(getTrainingMethod(), parentNodes);
    float parentEntropy = parentErrorFunction->error();
    float childEntroy = childErrorFunction->error();
    
    // The split is performed, create the childNodes with sufficient data
    // We now have to decide which nodes are split. Only nodes that have a sufficient number of training examples
    // in their child nodes are split. Then we have to determine the child nodes that are split next. Only child
    // nodes that have sufficiently much data and are not pure are split. 

    // We store the data count for each (virtual) child node here
    int* dataCounts = new int[childNodeCount];
    // Initialize the array
    for (int i = 0; i < childNodeCount; i++)
    {
        dataCounts[i] = 0;
    }

    // Compute the data count for all child nodes
    for (std::set<TrainingDAGNode::ptr>::iterator it = parentNodes.begin(); it != parentNodes.end(); ++it)
    {
        int leftNode = (*it)->getTempLeft();
        int rightNode = (*it)->getTempRight();

        dataCounts[leftNode] += (*it)->getLeftDataCount();
        dataCounts[rightNode] += (*it)->getRightDataCount();
    }

    // Check for every parent node if the split can be performed
    for (std::set<TrainingDAGNode::ptr>::iterator it = parentNodes.begin(); it != parentNodes.end(); ++it)
    {
        int leftNode = (*it)->getTempLeft();
        int rightNode = (*it)->getTempRight();

        if (dataCounts[leftNode] <= getMinChildSplitCount() || dataCounts[rightNode] <= getMinChildSplitCount())
        {
            dataCounts[leftNode] -= (*it)->getLeftDataCount();
            dataCounts[rightNode] -= (*it)->getRightDataCount();

            // The split cannot be performed
            (*it)->setTempLeft(-1);
            (*it)->setTempRight(-1);
        }
    }

    // Create the child nodes
    std::vector<TrainingDAGNode::ptr> childNodes(childNodeCount);
    for (int i = 0; i < childNodeCount; i++)
    {
        if (dataCounts[i] > 0)
        {
            childNodes[i] = TrainingDAGNode::Factory::create(this);
        }
    }

    // Assign the parent to their child nodes and set the training sets
    for (std::set<TrainingDAGNode::ptr>::iterator it = parentNodes.begin(); it != parentNodes.end(); ++it)
    {
        int leftNode = (*it)->getTempLeft();
        int rightNode = (*it)->getTempRight();

        // Skip this node if it is not split any further
        if (leftNode == -1) continue;

        // Assign the parent to the children
        (*it)->setLeft(childNodes[leftNode]);
        (*it)->setRight(childNodes[rightNode]);

        // Propagate the training set
        TrainingSet::ptr parentTrainingSet = (*it)->getTrainingSet();

        for (TrainingSet::iterator tit = parentTrainingSet->begin(); tit != parentTrainingSet->end(); ++tit)
        {
            // Determine whether or not this example belongs to the left or right child node
            if ((*(*tit)->getDataPoint())[(*it)->getFeatureID()] <= (*it)->getThreshold())
            {
                // Left child node
                childNodes[leftNode]->getTrainingSet()->push_back(*tit);
            }
            else
            {
                // Right child node
                childNodes[rightNode]->getTrainingSet()->push_back(*tit);
            }
        }
    }
    for (int i = 0; i < childNodeCount; i++)
    {
        // Does this child node exist?
        if (!childNodes[i])
        {
            // Nope
            continue;
        }

        // Select the class label and compute the class histogram for this child node
        childNodes[i]->updateHistogramAndLabel();
    }

    std::set<TrainingDAGNode::ptr> returnChildNodes;
    
    if (std::abs(parentEntropy - childEntroy) > 1e-10)
    {
        // Decide which nodes to split further
        for (int i = 0; i < childNodeCount; i++)
        {
            // Does this child node exist?
            if (!childNodes[i])
            {
                // Nope
                continue;
            }

            // Select the class label and compute the class histogram for this child node
            childNodes[i]->updateHistogramAndLabel();

            // Only split this node if it's not pure and has enough training data
            if (!TrainingUtil::histogramIsDirichlet(childNodes[i]->getClassHistogram())
                    && childNodes[i]->getTrainingSet()->size() > getMinSplitcount())
            {
                // This is a pure node, don't split it
                returnChildNodes.insert(childNodes[i]);
                continue;
            }
        }
    }
    delete[] dataCounts;

    return returnChildNodes;
}

TrainingDAGNode::ptr TrainingDAGNode::Factory::create(DAGTrainerPtr trainer)
{
    TrainingDAGNode::ptr node = new TrainingDAGNode(trainer);

    // Initialize the parent parameters
    DAGNode::Factory::init(node);

    node->trainingSet = TrainingSet::Factory::create();
    
    // Initialize the training parameters
    node->setTempLeft(0);
    node->setTempRight(0);
    node->setClassLabel(0);

    // Initialize the histograms
    node->setClassHistogram(ClassHistogram::Factory::createEmpty(trainer->getClassCount()));
    node->leftHistogram = ClassHistogram::Factory::createEmpty(trainer->getClassCount());
    node->rightHistogram = ClassHistogram::Factory::createEmpty(trainer->getClassCount());

    return node;
}

void TrainingDAGNode::resetLeftRightHistogram()
{
    // The left one becomes zero, the right one becomes the node histogram
    ClassHistogram::ptr nodeHistogram = getClassHistogram();

    for (int i = 0; i < trainer->getClassCount(); i++)
    {
        (*leftHistogram)[i] = 0;
        (*rightHistogram)[i] = (*nodeHistogram)[i];
    }
    leftDataCount = 0;
    rightDataCount = trainingSet->size();
}


void TrainingDAGNode::updateHistogramAndLabel()
{
    // Compute the histogram
    TrainingUtil::computHistogram(getClassHistogram(), getTrainingSet());
    // Get the best class label
    setClassLabel(TrainingUtil::histogramArgMax(getClassHistogram()));
}

bool TrainingDAGNode::findThreshold(AbstractErrorFunction::ptr error)
{
    // We need to save the current settings in order to restore them after optimization because we modify the object
    // in order to evaluate the error function
    ClassHistogram::ptr bestLeftHistogram = ClassHistogram::Factory::clone(leftHistogram);
    ClassHistogram::ptr bestRightHistogram = ClassHistogram::Factory::clone(rightHistogram);
    int bestLeftDataCount = leftDataCount;
    int bestRightDataCount = rightDataCount;
    int bestFeatureID = getFeatureID();
    double bestThreshold = getThreshold();
    
    // Compute the current error in order to find a better threshold
    float bestEntropy = error->error();
    float currentEntropy = 0;
    
    // Current threshold
    double threshold = 0;
    
    // Return flag to notify the calling optimizer whether or not we changed the threshold
    bool changed = false;
    
    // Iterate over all sampled features
    std::vector<int> sampledFeatures;
    trainer->getSampledFeatures(sampledFeatures);
    for (std::vector<int>::iterator fit = sampledFeatures.begin(); fit != sampledFeatures.end(); ++fit)
    {
        int f = *fit;
        
        // Sort the training set according to the current feature dimension
        TrainingExampleComparator compare(f);
        std::sort(trainingSet->begin(), trainingSet->end(), compare);
        
        // Initialize the virtual left/right histograms
        resetLeftRightHistogram();
        
        // Test all possible splits
        for (TrainingSet::iterator it = trainingSet->begin(); it != trainingSet->end() - 1; ++it)
        {
            // Choose the threshold as value between the two adjacent elements
            threshold = ((*it)->getDataPoint()->at(f) + (*(it + 1))->getDataPoint()->at(f))/2;
            
            // Update the histograms
            (*leftHistogram)[(*it)->getClassLabel()] += 1;
            (*rightHistogram)[(*it)->getClassLabel()] -= 1;
            
            leftDataCount++;
            rightDataCount--;
            
            // Get the current entropy
            currentEntropy = error->error();
            // Only accept the split if the entropy decreases and the threshold is not insignificant
            if (currentEntropy > bestEntropy && ( (*(it + 1))->getDataPoint()->at(f) - (*it)->getDataPoint()->at(f)) >= 1e-6)
            {
                // Select this feature and this threshold
                bestFeatureID = f;
                bestThreshold = threshold;
                for (int i = 0; i < trainer->getClassCount(); i++)
                {
                    (*bestLeftHistogram)[i] = (*leftHistogram)[i];
                    (*bestRightHistogram)[i] = (*rightHistogram)[i];
                }
                bestLeftDataCount = leftDataCount;
                bestRightDataCount = rightDataCount;
                bestEntropy = currentEntropy;
                changed = true;
            }
        }
    }
    
    // Restore the arg min settings
    leftHistogram = bestLeftHistogram;
    rightHistogram = bestRightHistogram;
    leftDataCount = bestLeftDataCount;
    rightDataCount = bestRightDataCount;
    setFeatureID(bestFeatureID);
    setThreshold(bestThreshold);
    
    return changed;
}

bool TrainingDAGNode::findChildNodeAssignment(AbstractErrorFunction::ptr error, int childNodeCount)
{
    // Save the currently best settings
    int selectedLeft = tempLeft;
    int selectedRight = tempRight;
    
    float bestEntropy = error->error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int cLeft = 0; cLeft < childNodeCount; cLeft++)
    {
        for (int cRight = 0; cRight < childNodeCount; cRight++)
        {
            // Do not assign both pointers to the same node
            // This doesn't make any sense
            if (cLeft == cRight) continue;
            
            // Test this assignment
            setTempLeft(cLeft);
            setTempRight(cRight);
            
            // Get the error
            currentEntropy = error->error();
            
            // Is this better?
            if (currentEntropy > bestEntropy)
            {
                // it is
                selectedLeft = cLeft;
                selectedRight = cRight;
                bestEntropy = currentEntropy;
                changed = true;
            }
        }
    }
    
    // Restore the arg min setting
    setTempLeft(selectedLeft);
    setTempRight(selectedRight);
    
    return changed;
}


bool TrainingDAGNode::findLeftChildNodeAssignment(AbstractErrorFunction::ptr error, int childNodeCount)
{
    // Save the currently best settings
    int selectedLeft = tempLeft;
    int selectedRight = tempRight;
    
    float bestEntropy = error->error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int cLeft = 0; cLeft < childNodeCount; cLeft++)
    {
        // Do not assign both pointers to the same node
        // This doesn't make any sense
        if (cLeft == selectedRight) continue;

        // Test this assignment
        setTempLeft(cLeft);

        // Get the error
        currentEntropy = error->error();

        // Is this better?
        if (currentEntropy > bestEntropy)
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

bool TrainingDAGNode::findRightChildNodeAssignment(AbstractErrorFunction::ptr error, int childNodeCount)
{
    // Save the currently best settings
    int selectedLeft = tempLeft;
    int selectedRight = tempRight;
    
    float bestEntropy = error->error();
    float currentEntropy = 0;
    bool changed = false;
    
    // Test all possible assignments
    for (int cRight = 0; cRight < childNodeCount; cRight++)
    {
        // Do not assign both pointers to the same node
        // This doesn't make any sense
        if (selectedLeft == cRight) continue;

        // Test this assignment
        setTempRight(cRight);

        // Get the error
        currentEntropy = error->error();

        // Is this better?
        if (currentEntropy > bestEntropy)
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

TrainingSet::ptr TrainingSet::Factory::createBySampling(TrainingSet::ptr _trainingSet, int n)
{
    return _trainingSet;
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
    
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(lineCount);
    
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    std::vector< std::string > row;
    std::string line;

    if (_verboseMode)
    {
        std::cout << "Loading training set from " << _fileName << std::endl;
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
        
        // Load the training example to the training set
        TrainingExample::ptr example = TrainingExample::Factory::createFromFileRow(row);
        trainingSet->push_back(example);
    }
    std::cout << "Training set loaded. Number of examples: " << trainingSet->size() << std::endl ;
    
    return trainingSet;
}

DAGTrainer::ptr DAGTrainer::Factory::createFromJungleTrainer(JungleTrainer::ptr _jungleTrainer, TrainingSet::ptr _trainingSet)
{
    DAGTrainer::ptr result = createForTraingSet(_trainingSet);
    
    // Transfer all parameter
    result->setMaxDepth(_jungleTrainer->getMaxDepth());
    result->setMaxWidth(_jungleTrainer->getMaxWidth());
    result->setMinSpliCount(_jungleTrainer->getMinSplitcount());
    result->setMinChildSplitCount(_jungleTrainer->getMinChildSplitCount());
    result->setTrainingMethod(_jungleTrainer->getTrainingMethod());
    result->setVerboseMode(_jungleTrainer->getVerboseMode());
    result->setNumFeatureSamples(_jungleTrainer->getNumFeatureSamples());
    
    return result;
}


void AbstractTrainer::Factory::init(AbstractTrainer::ptr _trainer)
{
    _trainer->maxDepth = 256;
    _trainer->maxWidth = 1024;
    _trainer->minSplitCount = 3;
    _trainer->minChildSplitCount = 3;
    _trainer->trainingMethod = AbstractErrorFunction::error_entropy;
    _trainer->verboseMode = true;
    // -1 means that the number of features to sample will be determined automatically
    _trainer->numFeatureSamples = -1;
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
        numTrainingSamples = std::min(static_cast<int>(trainingSet->size()), static_cast<int>(std::floor(trainingSet->size() * 5 / static_cast<double>(numDAGs))));
    }
    
    Jungle::ptr jungle = Jungle::Factory::create();
    
    if (getVerboseMode())
    {
        printf("Start training\n");
        printf("Number of training examples: %d\n", static_cast<int>(trainingSet->size()));
        printf("Number of examples per DAG: %d\n", getNumTrainingSamples());
        printf("Number of DAGs to train: %d\n", getNumDAGs());
    }
    
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(getNumDAGs());
    if (getVerboseMode())
    {
        progressBar->update(0);
        std::cout.flush();
    }
    for (int i = 0; i < numDAGs; i++)
    {
        // Create a training set for each DAG by sampling from the given training set
        TrainingSet::ptr sampledSet = TrainingSet::Factory::createBySampling(trainingSet, numTrainingSamples);
        
        DAGTrainer::ptr trainer = DAGTrainer::Factory::createFromJungleTrainer(this, sampledSet);
        jungle->getDAGs().insert(trainer->train());
        
        if (getVerboseMode())
        {
            progressBar->update();
            std::cout.flush();
        }
    }
    
    return jungle;
}

AbstractErrorFunction::ptr AbstractErrorFunction::Factory::createNodeErrorFunction(char _criteria, TrainingDAGNode::ptr node)
{
    // Determine the error measure to use
    switch (_criteria)
    {
        case AbstractErrorFunction::error_entropy:
            return AbstractEntropyErrorFunction::Factory::createNodeErrorFunction(node);
            break;

        default:
            throw RuntimeException("Unknown error measure.");
            break;
    }
}

AbstractErrorFunction::ptr AbstractErrorFunction::Factory::createNodeRowErrorFunction(char _criteria, std::set<TrainingDAGNode::ptr>& _nodes)
{
    // Determine the error measure to use
    switch (_criteria)
    {
        case AbstractErrorFunction::error_entropy:
            return AbstractEntropyErrorFunction::Factory::createNodeRowErrorFunction(_nodes);
            break;

        default:
            throw RuntimeException("Unknown error measure.");
            break;
    }
}

AbstractErrorFunction::ptr AbstractErrorFunction::Factory::createChildNodeRowErrorFunction(char _criteria, std::set<TrainingDAGNode::ptr>& _nodes, int _childNodeCount)
{
    // Determine the error measure to use
    switch (_criteria)
    {
        case AbstractErrorFunction::error_entropy:
            return AbstractEntropyErrorFunction::Factory::createChildNodeRowErrorFunction(_nodes, _childNodeCount);
            break;

        default:
            throw RuntimeException("Unknown error measure.");
            break;
    }
}

AbstractErrorFunction::ptr AbstractErrorFunction::Factory::createNewChildNodeRowErrorFunction(char _criteria, std::set<TrainingDAGNode::ptr>& _nodes, TrainingDAGNode* _parent)
{
    // Determine the error measure to use
    switch (_criteria)
    {
        case AbstractErrorFunction::error_entropy:
            return AbstractEntropyErrorFunction::Factory::createNewChildNodeRowErrorFunction(_nodes, _parent);
            break;

        default:
            throw RuntimeException("Unknown error measure.");
            break;
    }
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createNodeErrorFunction(TrainingDAGNode::ptr node)
{
    AbstractErrorFunctionPtr errorFunction = new NodeEntropyErrorFunction(node);
    return errorFunction;
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createNodeRowErrorFunction(std::set<TrainingDAGNode::ptr>& _nodes)
{
    AbstractErrorFunctionPtr errorFunction = new RowEntropyErrorFunction(&_nodes);
    return errorFunction;
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createChildNodeRowErrorFunction(std::set<TrainingDAGNode::ptr>& _nodes, int _childNodeCount)
{
    AbstractErrorFunctionPtr errorFunction = new ChildRowEntropyErrorFunction(&_nodes, _childNodeCount);
    return errorFunction;
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createNewChildNodeRowErrorFunction(std::set<TrainingDAGNode::ptr>& _nodes, TrainingDAGNode::ptr _parent)
{
    AbstractErrorFunctionPtr errorFunction = new NewChildRowEntropyErrorFunction(&_nodes, _parent);
    return errorFunction;
}

float TrainingStatistics::trainingError(Jungle::ptr _jungle, TrainingSet::ptr _trainingSet)
{
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(_trainingSet->size());
   
    if (getVerboseMode())
    {
        std::cout << "Calculate training error" << std::endl;
    }
    
    // Calculate the training error
    float error = 0;
    for (TrainingSet::iterator iter = _trainingSet->begin(); iter != _trainingSet->end(); ++iter)
    {
        // Update the progress bar
        if (getVerboseMode())
        {
            progressBar->update();
        }
        
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
    
    if (getVerboseMode() || 1)
    {
        printf("Training error: %2.4f\n", error);
    }
    
    return error;
}


void NewChildRowEntropyErrorFunction::initHistograms()
{
    int classCount = (*row->begin())->getClassHistogram()->size();

    leftHistogram = new int[classCount];
    rightHistogram = new int[classCount];
    cleftHistogram = new int[classCount];
    crightHistogram = new int[classCount];

    // Initialize the histograms
    for (int i = 0; i < classCount; i++)
    {
        leftHistogram[i] = 0;
        rightHistogram[i] = 0;
    }

    // We store the data count for each (virtual) child node here
    leftDataCount = 0;
    rightDataCount = 0;

    // Compute the histograms for all child nodes
    for (std::set<TrainingDAGNode::ptr>::iterator it = row->begin(); it != row->end(); ++it)
    {
        // Skip the parent node
        if (*it == parent) continue;
        
        int leftNode = (*it)->getTempLeft();
        ClassHistogram::ptr _leftHistogram = (*it)->getLeftHistogram();
        int rightNode = (*it)->getTempRight();
        ClassHistogram::ptr _rightHistogram = (*it)->getRightHistogram();

        if (leftNode == parent->getTempLeft())
        {
            // Add the values to the histogram
            for (int i = 0; i < classCount; i++)
            {
                leftHistogram[i] += (*_leftHistogram)[i];
            }
            leftDataCount += (*it)->getLeftDataCount();
        }
        else if (leftNode == parent->getTempRight())
        {
            // Add the values to the histogram
            for (int i = 0; i < classCount; i++)
            {
                rightHistogram[i] += (*_leftHistogram)[i];
            }
            rightDataCount += (*it)->getLeftDataCount();
        }

        if (rightNode == parent->getTempLeft())
        {
            // Add the values to the histogram
            for (int i = 0; i < classCount; i++)
            {
                leftHistogram[i] += (*_rightHistogram)[i];
            }
            leftDataCount += (*it)->getRightDataCount();
        }
        else if (leftNode == parent->getTempRight())
        {
            // Add the values to the histogram
            for (int i = 0; i < classCount; i++)
            {
                rightHistogram[i] += (*_rightHistogram)[i];
            }
            rightDataCount += (*it)->getRightDataCount();
        }
    }
}

float NewChildRowEntropyErrorFunction::error() const
{
    float result = 0.;
    int classCount = (*row->begin())->getClassHistogram()->size();


    ClassHistogram::ptr _leftHistogram = parent->getLeftHistogram();
    ClassHistogram::ptr _rightHistogram = parent->getRightHistogram();
/*
    printf("Parent (left):    ");
    for (int i = 0; i < classCount; i++)
    {
        printf("%6d: %6d ", i, _leftHistogram->at(i));
    }
    printf("\n");

    printf("Parent (right):   ");
    for (int i = 0; i < classCount; i++)
    {
        printf("%6d: %6d ", i, _rightHistogram->at(i));
    }
    printf("\n");
    printf("Virtual  (left):  ");
    for (int i = 0; i < classCount; i++)
    {
        printf("%6d: %6d ", i, leftHistogram[i]);
    }
    printf("\n");

    printf("Virtual  (right): ");
    for (int i = 0; i < classCount; i++)
    {
        printf("%6d: %6d ", i, rightHistogram[i]);
    }
    printf("\n\n");
*/
    // Initialize the histograms
    for (int i = 0; i < classCount; i++)
    {
        cleftHistogram[i] = leftHistogram[i] + _leftHistogram->at(i);
        crightHistogram[i] = rightHistogram[i] + _rightHistogram->at(i);
    }

    // We store the data count for each (virtual) child node here
    int cleftDataCount = leftDataCount + parent->getLeftDataCount();
    int crightDataCount = rightDataCount + parent->getRightDataCount();

    float dataCount = cleftDataCount + crightDataCount;
    result = cleftDataCount/dataCount * entropy(cleftHistogram, classCount);
    result += crightDataCount/dataCount * entropy(crightHistogram, classCount);

    return result;
}