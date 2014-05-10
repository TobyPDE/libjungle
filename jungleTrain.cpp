
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include "jungleTrain.h"

using namespace decision_jungle;

int main(int a, char** b)
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
    TrainingDAGNode::ptr root = TrainingDAGNode::Factory::create(shared_from_this());
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
    float parentEntropy = childErrorFunction->error();
    // FIXME
    
    // Adjust the thresholds and child assignments until nothing changes anymore
    bool change = false;
    do
    {
        change = false;
        for (std::set<TrainingDAGNode::ptr>::iterator iter = parentNodes.begin(); iter != parentNodes.end(); ++iter)
        {
            // Find the new optimal threshold
            if ((*iter)->findThreshold(childErrorFunction))
            {
                change = true;
            }
        }
        
        for (std::set<TrainingDAGNode::ptr>::iterator iter = parentNodes.begin(); iter != parentNodes.end(); ++iter)
        {
            // Find the new optimal child assignment
            if ((*iter)->findChildNodeAssignment(childErrorFunction, childNodeCount))
            {
                change = true;
            }
        }
    }
    while (change);
    
    // Determine whether or not the row training shall be performed
    // Get the entropy of the parent row in order to determine whether or not the split shall be performed
    AbstractErrorFunction::ptr parentErrorFunction = AbstractErrorFunction::Factory::createNodeRowErrorFunction(getTrainingMethod(), parentNodes);
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
            childNodes[i] = TrainingDAGNode::Factory::create(shared_from_this());
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
            if (!TrainingUtil::histogramIsAlmostDirichlet(childNodes[i]->getClassHistogram(), getMinSplitcount())
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
    TrainingDAGNode::ptr node(new TrainingDAGNode(trainer));

    // Initialize the parent parameters
    DAGNode::Factory::init(node);

    node->trainingSet = TrainingSet::Factory::create();
    
    // Initialize the training parameters
    node->setTempLeft(0);
    node->setTempRight(0);

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
    _trainer->maxWidth = 256;
    _trainer->minSplitCount = 30;
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
    
    printf("Number of training examples: %d\n", static_cast<int>(trainingSet->size()));
    printf("Number of examples per DAG: %d\n", getNumTrainingSamples());
    
    // Create a progress bar
    int progressWidth = 20;
    float progress = 0.;
    
    for (int i = 0; i < numDAGs; i++)
    {
        if (numDAGs > 1)
        {
            progress = i/static_cast<float>(numDAGs - 1);
        }
        else
        {
            progress = 1;
        }
        
        printf("\rTraining DAGs [");
        for (int j = 0; j < progressWidth; j++)
        {
            if (j <= progress*progressWidth)
            {
                printf("*");
            }
            else
            {
                printf(" ");
            }
        }
        printf("] %4d/%4d (%2.1f%%)", i+1, numDAGs, progress*100);
        
        // Create a training set for each DAG by sampling from the given training set
        TrainingSet::ptr sampledSet = TrainingSet::Factory::createBySampling(trainingSet, numTrainingSamples);
        
        DAGTrainer::ptr trainer = DAGTrainer::Factory::createFromJungleTrainer(shared_from_this(), sampledSet);
        jungle->getDAGs().insert(trainer->train());
    }
    printf("\n");
    
    
    // Calculate the training error
    float error = 0;
    for (TrainingSet::iterator iter = trainingSet->begin(); iter != trainingSet->end(); ++iter)
    {
        if ((*iter)->getClassLabel() != jungle->predict((*iter)->getDataPoint())->getClassLabel())
        {
            error++;
        }
    }
    printf("Training error: %2.4f\n", error/static_cast<float>(trainingSet->size()));
    
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

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createNodeErrorFunction(TrainingDAGNode::ptr node)
{
    AbstractErrorFunctionPtr errorFunction(new NodeEntropyErrorFunction(node));
    return errorFunction;
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createNodeRowErrorFunction(std::set<TrainingDAGNode::ptr>& _nodes)
{
    AbstractErrorFunctionPtr errorFunction(new RowEntropyErrorFunction(&_nodes));
    return errorFunction;
}

AbstractErrorFunction::ptr AbstractEntropyErrorFunction::Factory::createChildNodeRowErrorFunction(std::set<TrainingDAGNode::ptr>& _nodes, int _childNodeCount)
{
    AbstractErrorFunctionPtr errorFunction(new ChildRowEntropyErrorFunction(&_nodes, _childNodeCount));
    return errorFunction;
}

