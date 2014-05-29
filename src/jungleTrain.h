#ifndef JUNGLE_TRAIN_H
#define JUNGLE_TRAIN_H

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <cmath>

#include "jungle.h"

/**
 * This file everything that is important for training a jungle
 * 
 * @author Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
 * @version 1.0
 */
namespace decision_jungle {
    
    /**
     * Forward declarations
     */
    class TrainingDAGNode;
    class DAGTrainer;
    typedef DAGTrainer* DAGTrainerPtr;
    class TrainingExample;
    class TrainingSet;
    class JungleTrainer;
    typedef JungleTrainer* JungleTrainerPtr;
    typedef std::vector< std::vector<float> > Matrix;
    typedef std::vector< TrainingDAGNode* > NodeRow;
    
    /**
     * A training example consists of a data point and a class label
     */
    class TrainingExample {
    private:
        /**
         * The data point/feature vector corresponding to this training example
         */
        DataPoint::ptr dataPoint;
        
        /**
         * The class label for this example
         */
        ClassLabel classLabel;
        
    public:
        typedef TrainingExample self;
        typedef self* ptr;
        
        /**
         * Default constructor
         */
        TrainingExample(DataPoint::ptr _dataPoint, ClassLabel _classLabel) : 
                dataPoint(_dataPoint), classLabel(_classLabel) {}
        
        /**
         * Copy constructor
         */
        TrainingExample(const TrainingExample& other) : dataPoint(other.dataPoint), classLabel(other.classLabel) {}
        
        /**
         * Assignment operator
         */
        TrainingExample & operator= (const TrainingExample &other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                dataPoint = other.dataPoint;
                classLabel = other.classLabel;
            }
            return *this;
        }
        
        virtual ~TrainingExample()\
        {
            delete dataPoint;
        }
        
        /**
         * Returns the data point
         * 
         * @return Data point/feature vector
         */
        DataPoint::ptr getDataPoint()
        {
            return dataPoint;
        }
        
        /**
         * Returns the class label
         * 
         * @return corresponding class label
         */
        ClassLabel getClassLabel()
        {
            return classLabel;
        }
        
        /**
         * A factory class for "TrainingExample"
         */
        class Factory {
        public:
            /**
             * Creates a new training example from a class label and a feature vector
             * 
             * @param _dataPoint The feature vector
             * @param _classLabel The corresponding class label
             */
            static TrainingExample::ptr create(DataPoint::ptr _dataPoint, const ClassLabel _classLabel)
            {
                TrainingExample::ptr result = new TrainingExample(_dataPoint, _classLabel);
                return result;
            }
            
            /**
             * Creates a training example of class _classLabel and a zero initialized feature vector of dimension dim
             * 
             * @param _dim The dimension of the feature vector
             * @param _classLabel The class label
             */
            static TrainingExample::ptr createZeroInitialized(int _dim, ClassLabel _classLabel)
            {
                // Create a new data point
                DataPoint::ptr dataPoint = DataPoint::Factory::createZeroInitialized(_dim);
                return Factory::create(dataPoint, _classLabel);
            }
            
            /**
             * Creates a new data point from a row from a data set
             * 
             * @param _row The vector of strings representing the training example
             * @return The created training example
             */
            static TrainingExample::ptr createFromFileRow(const std::vector<std::string> & _row);
        };
    };
    
    /**
     * A training set consists of several training examples
     */
    class TrainingSet {
    public:
        // We use a vector because we need to sort the training set during training
        typedef std::vector<TrainingExample::ptr> self;
        typedef self::iterator iterator;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Deletes all data points in a training set
         */
        static void freeTrainingExamples(TrainingSet::ptr _trainingSet)
        {
            for (TrainingSet::iterator it = _trainingSet->begin(); it != _trainingSet->end(); ++it)
            {
                delete *it;
            }
            _trainingSet->erase(_trainingSet->begin(), _trainingSet->end());
        }
        
        /**
         * A factory class for training sets
         */
        class Factory {
        public:
            /**
             * Creates a new blank training set
             */
            static TrainingSet::ptr create() 
            {
                TrainingSet::ptr result(new self());
                return result;
            }
            
            /**
             * Creates a new training set by randomly samping n elements with replacement from the given set
             * 
             * @param _trainingSet Given training set
             * @param n The number of samples to draw
             * @return sampled set
             */
            static TrainingSet::ptr createBySampling(TrainingSet::ptr _trainingSet, int n);
            
            /**
             * Loads a training set from a file
             * 
             * @param _fileName The filename
             * @param _verboseMode
             * @return The loaded training set
             */
            static TrainingSet::ptr createFromFile(const std::string & _fileName, bool _verboseMode);
        };
    };
    
    /**
     * This version of a DAG node is used during training. It some additional information such as a list of data points
     * at this node and provides some additional functions. 
     */
    class TrainingDAGNode : public DAGNode {
    private:
        /**
         * This is the list of training examples at this node
         */
        TrainingSet::ptr trainingSet;
        
        /**
         * This is the class distribution at the left child node if there were no other nodes linking to this
         * node
         */
        ClassHistogram leftHistogram;
        
        /**
         * This is the class distribution at the right child node if there were no other nodes linking to this
         * node
         */
        ClassHistogram rightHistogram;
        
        /**
         * Reference to the trainer
         */
        DAGTrainerPtr trainer;
        
        /**
         * True if the node is pure
         */
        bool pure;
        
    public:
        typedef TrainingDAGNode self;
        typedef self* ptr;
        
        /**
         * Computes the left and right histograms
         */
        void updateLeftRightHistogram();
        
        /**
         * Resets the left and right histograms
         */
        void resetLeftRightHistogram();
        
        /**
         * Default constructor
         * @param _trainer The corresponding trainer instance
         */
        TrainingDAGNode(DAGTrainerPtr _trainer) : trainer(_trainer) { }
        
        /**
         * Copy constructor
         */
        TrainingDAGNode(const TrainingDAGNode& other) : 
                trainingSet(other.trainingSet), 
                // FIXME: The histograms should be copied
                leftHistogram(other.leftHistogram), 
                rightHistogram(other.rightHistogram), 
                pure(false) {}
        
        /**
         * Assignment operator
         */
        TrainingDAGNode & operator=(const TrainingDAGNode & other)
        {
            // Prevent self references
            if (this != &other)
            {
                trainingSet = other.trainingSet;
                leftHistogram = other.leftHistogram;
                rightHistogram = other.rightHistogram;
            }
            
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~TrainingDAGNode() {}
        
        /**
         * Selects the best class label and computes the class histogram based on the current training set
         */
        void updateHistogramAndLabel();
        
        /**
         * Returns the training set
         * 
         * @return the training set at this node
         */
        TrainingSet::ptr getTrainingSet()
        {
            return trainingSet;
        }
        
        /**
         * Returns a reference to the left histogram
         * 
         * @return Reference to left histogram
         */
        ClassHistogram* getLeftHistogram()
        {
            return & leftHistogram;
        }
        
        /**
         * Returns a reference to the right histogram
         * 
         * @return Reference to left histogram
         */
        ClassHistogram* getRightHistogram()
        {
            return & rightHistogram;
        }
        
        /**
         * Returns whether of not the node is pure
         * 
         * @return true if the node is pure
         */
        bool isPure()
        {
            return pure;
        }
        
        /**
         * Finds an optimal threshold based on the provided error function
         * 
         * @param error An error function to measure the entropy of the current setting
         * @return true if the threshold was changed.
         */
        bool findThreshold(NodeRow & parentNodes);
        
        /**
         * Finds an optimal child node assignment based on the given error measure
         * 
         * @param error An error function to measure the entropy of the current setting
         * @return true if the assignment was changed.
         */
        bool findCoherentChildNodeAssignment(NodeRow & parentNodes, int childNodeCount);
        bool findLeftChildNodeAssignment(NodeRow & parentNodes, int childNodeCount);
        bool findRightChildNodeAssignment(NodeRow & parentNodes, int childNodeCount);
        
        /**
         * Factory class for these training nodes
         */
        class Factory : public DAGNode::Factory {
        public:
            /**
             * Creates a new blank node for a trainer
             */
            static TrainingDAGNode::ptr create(DAGTrainerPtr trainer);
        };
    };
    
    /**
     * This class contains all the field a DAG and a jungle trainer have in common
     */
    class AbstractTrainer {
    private:
        /**
         * Number of features to sample
         */
        int numFeatureSamples;
        
        /**
         * The maximum depth of the DAG
         */
        int maxDepth;
        
        /**
         * The maximum width
         */
        int maxWidth;
        
        /**
         * Verbose more
         */
        bool verboseMode;
        
        /**
         * Whether of not bagging shall be used during training
         */
        bool useBagging;
        
        /**
         * The maximum number of iterations at each level
         */
        int maxLevelIterations;
        
        /**
         * True, if stochastic threshold estimation shall be used
         */
        bool useStochasticThreshold;
        
        /**
         * True, if the child node assignment shall be perform stochastically
         */
        bool useStochasticChildNodeAssignment;
        
        /**
         * The validation level
         */
        int validationLevel;
        
        /**
         * The validation set
         */
        TrainingSet::ptr validationSet;
        
    protected:
        /**
         * Validates all parameters and throws an exception is some parameters are invalid
         * 
         * @throws ConfigurationException If the configuration is invalid
         */
        virtual void validateParameters() throw(ConfigurationException);
        
    public:
        typedef AbstractTrainer self;
        typedef self* ptr;
        
        /**
         * Sets validationLevel
         */
        void setValidationLevel(int _validationLevel)
        {
            validationLevel = _validationLevel;
        }
        
        /**
         * Returns validationLevel
         * 
         * @return validationLevel
         */
        int getValidationLevel()
        {
            return validationLevel;
        }
        
        /**
         * Sets the validation set
         */
        void setValidationSet(TrainingSet::ptr _validationSet)
        {
            validationSet = _validationSet;
        }
        
        /**
         * Returns the validation set
         */
        TrainingSet::ptr getValidationSet()
        {
            return validationSet;
        }
        
        /**
         * Sets useBagging
         * 
         * @param _useBagging
         */
        void setUseBagging(bool _useBagging)
        {
            useBagging = _useBagging;
        }
        
        /**
         * Returns useBagging
         * 
         * @return useBagging
         */
        bool getUseBagging()
        {
            return useBagging;
        }
        
        /**
         * Sets maxLevelIterations
         * 
         * @param _maxLevelIterations
         */
        void setMaxLevelIterations(int _maxLevelIterations)
        {
            maxLevelIterations = _maxLevelIterations;
        }
        
        /**
         * Returns maxLevelIterations
         * 
         * @return maxLevelIterations
         */
        int getMaxLevelIterations()
        {
            return maxLevelIterations;
        }
        
        /**
         * Sets useStochasticThreshold
         * 
         * @param _useStochasticThreshold
         */
        void setUseStochasticThreshold(bool _useStochasticThreshold)
        {
            useStochasticThreshold = _useStochasticThreshold;
        }
        
        /**
         * Returns useStochasticThreshold
         * 
         * @return useStochasticThreshold
         */
        bool getUseStochasticThreshold()
        {
            return useStochasticThreshold;
        }
        
        /**
         * Sets useStochasticChildNodeAssignment
         * 
         * @param _useStochasticChildNodeAssignment
         */
        void setUseStochasticChildNodeAssignment(bool _useStochasticChildNodeAssignment)
        {
            useStochasticChildNodeAssignment = _useStochasticChildNodeAssignment;
        }
        
        /**
         * Returns useStochasticChildNodeAssignment
         * 
         * @return useStochasticChildNodeAssignment
         */
        bool getUseStochasticChildNodeAssignment()
        {
            return useStochasticChildNodeAssignment;
        }
        
        /**
         * Sets the status of verbose mode
         * 
         * @param _verboseMode
         */
        void setVerboseMode(bool _verboseMode)
        {
            verboseMode = _verboseMode;
        }
        
        /**
         * Returns the current status of verbose mode
         * 
         * @return true if verbose mode is on
         */
        bool getVerboseMode()
        {
            return verboseMode;
        }

        /**
         * Sets the max depth
         * 
         * @param _maxDepth new max depth
         */
        void setMaxDepth(int _maxDepth)
        {
            maxDepth = _maxDepth;
        }
        
        /**
         * Returns the max depth
         * 
         * @return max depth
         */
        int getMaxDepth()
        {
            return maxDepth;
        }
        
        /**
         * Sets the max width
         * 
         * @param _maxWidth new max width
         */
        void setMaxWidth(int _maxWidth)
        {
            maxWidth = _maxWidth;
        }
        
        /**
         * Returns the max width
         * 
         * @return max width
         */
        int getMaxWidth()
        {
            return maxWidth;
        }
        
        /**
         * Sets the number of features to sample
         * 
         * @param _numFeatureSamples The number of features to sample
         */
        void setNumFeatureSamples(int _numFeatureSamples)
        {
            numFeatureSamples = _numFeatureSamples;
        }
        
        /**
         * Returns the number of features to sample
         * 
         * @return Number of features to sample
         */
        int getNumFeatureSamples()
        {
            return numFeatureSamples;
        }
        
        /**
         * Outputs a message in verbose mode
         * 
         * @param _message
         */
        void verboseMessage(const std::string & _message)
        {
            if (verboseMode)
            {
                std::cout << _message << std::endl;
            }
        }
        
        /**
         * Base factory for all trainer factories
         */
        class Factory {
        protected:
            /**
             * Initializes the base parameters
             */
            static void init(ptr);
        };
    };
    
    /**
     * Trains a single DAG on a training set using the LSearch algorithm proposed in [1]. The width is controlled as
     * min(2^l, maxWidth) where l is the current level
     */
    class DAGTrainer : public AbstractTrainer {
    private:
        /**
         * The training set that is used for this DAG
         */
        TrainingSet::ptr trainingSet;

        /**
         * The sampled features for this DAG
         */
        std::vector<int> sampledFeatures;
        
        /**
         * The used feature dimension. These values are set by validateParameters()
         */
        int featureDimension;
        
        /**
         * The number of classes. These values are set by validateParameters()
         */
        int classCount;
        
        /**
         * Validates all parameters and throws an exception is some parameters are invalid
         * 
         * @throws ConfigurationException If the configuration is invalid
         */
        void validateParameters() throw(ConfigurationException);
        
        /**
         * Trains a single level of the DAG
         * 
         * @param parentNodes The set of parent nodes
         * @param childNodeCount The number of child nodes
         */
        NodeRow trainLevel(NodeRow &parentNodes, int childNodeCount);
        
    public:
        typedef DAGTrainer self;
        typedef self* ptr;
        
        virtual ~DAGTrainer() {}
        
        /**
         * Returns the feature dimension
         * 
         * @return Feature dimension
         */
        int getFeatureDimension()
        {
            return featureDimension;
        }
        
        /**
         * Returns the class count
         * 
         * @return class count
         */
        int getClassCount()
        {
            return classCount;
        }
        
        /**
         * Returns a list of sampled features
         * 
         * @return List of sampled features
         */
        void getSampledFeatures(std::vector<int> & sampledFeature);
        
        /**
         * Trains the DAG
         * 
         * @throws ConfigurationException If the configuration is invalid
         * @throws RuntimeException If training fails unexpectedly
         */
        TrainingDAGNode::ptr train() throw(ConfigurationException, RuntimeException);
        
        /**
         * Factory class for the trainer
         */
        class Factory : public AbstractTrainer::Factory {
        private:
            
        public:
            /**
             * Creates a new DAG trainer for a specific training set
             * 
             * @param _trainingSet the set to train on
             * @return new trainer instance
             */
            static DAGTrainer::ptr createForTraingSet(TrainingSet::ptr _trainingSet)
            {
                DAGTrainer::ptr trainer = new DAGTrainer();
                trainer->trainingSet = _trainingSet;
                
                // Initialize the trainer with the default parameters
                init(trainer);
                
                return trainer;
            }
            
            /**
             * Creates a new DAG trainer from the settings of a jungle trainer
             * 
             * @param _jungleTrainer The jungle trainer
             * @param _trainingSet The set to train on
             * @return new trainer instance
             */
            static DAGTrainer::ptr createFromJungleTrainer(JungleTrainerPtr _jungleTrainer, TrainingSet::ptr _trainingSet);
        };
    };
    
    /**
     * This is a trainer for decision jungle
     */
    class JungleTrainer : public AbstractTrainer {
    private:
        /**
         * Number of DAGs to train
         */
        int numDAGs;
        
        /**
         * Number of training samples per DAG
         */
        int numTrainingSamples;
    public:
        
        typedef JungleTrainer self;
        typedef self* ptr;
        
        virtual ~JungleTrainer() {}
        
        /**
         * Sets the number of training samples per DAG
         * -1: The number is determined based on the training set
         * 
         * @param _numTrainingSamples 
         */
        void setNumTrainingSamples(int _numTrainingSamples)
        {
            numTrainingSamples = _numTrainingSamples;
        }
        
        /**
         * Returns the number of training samples per DAG
         * 
         * @return Number of training samples per DAG
         */
        int getNumTrainingSamples()
        {
            return numTrainingSamples;
        }
        
        /**
         * Sets the number of dags to train
         * 
         * @param _numDAGs
         */
        void setNumDAGs(int _numDAGs)
        {
            numDAGs = _numDAGs;
        }
        
        /**
         * Returns the number of DAGs to train
         * 
         * @return number of dags to train
         */
        int getNumDAGs()
        {
            return numDAGs;
        }
        
        /**
         * Trains the DAG
         * 
         * @param trainingSet the set to train on
         * @throws ConfigurationException If the configuration is invalid
         * @throws RuntimeException If training fails unexpectedly
         */
        Jungle::ptr train(TrainingSet::ptr trainingSet) throw(ConfigurationException, RuntimeException);

        /**
         * Factory class for the trainer
         */
        class Factory : public AbstractTrainer::Factory {
        private:
            /**
             * Initializes a trainer with the default parameters
             * @param _trainer
             */
            static void init(JungleTrainer::ptr _trainer);
            
        public:
            /**
             * Creates a new jungle trainer for a specific training set
             * 
             * @return new trainer instance
             */
            static JungleTrainer::ptr create()
            {
                JungleTrainer::ptr trainer(new JungleTrainer());
                
                // Initialize the trainer with the default parameters
                init(trainer);
                
                return trainer;
            }
        };
    };
    
    /**
     * Calculates the entropy for an entire row of nodes
     */
    class RowEntropyErrorFunction {
    private:
        /**
         * The row of nodes
         */
        NodeRow & row;
        
    public:
        /**
         * Default constructor
         * @return 
         */
        RowEntropyErrorFunction (NodeRow & _row) : row(_row) {}
        
        /**
         * Copy constructor
         */
        RowEntropyErrorFunction (const RowEntropyErrorFunction & other) : row(other.row) {}
        
        /**
         * Assignment operator
         */
        RowEntropyErrorFunction & operator=(const RowEntropyErrorFunction & other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                row = other.row;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~RowEntropyErrorFunction() {}
        
        /**
         * Calculates the error if we split. This function expects the local histograms to be already computed.
         */
        float error() const
        {
            float result = 0.;

            // Determine the complete data count over all nodes
            int dataCount = 0;
            for (NodeRow::iterator it = row.begin(); it != row.end(); ++it)
            {
                dataCount += static_cast<int>((*it)->getTrainingSet()->size());
            }
            
            for (NodeRow::iterator it = row.begin(); it != row.end(); ++it)
            {
                 result += static_cast<float>( (*it)->getTrainingSet()->size()) / static_cast<float>(dataCount) * (*it)->getClassHistogram()->entropy();
            }

            return result;
        }
    };
    
    /**
     * Calculates the entropy error based on a child row
     */
    class ChildRowEntropyErrorFunction {
    private:
        /**
         * The row of nodes
         */
        NodeRow & row;
        /**
         * Number of child nodes
         */
        int childNodeCount;
        
    public:
        /**
         * Default constructor
         * @return 
         */
        ChildRowEntropyErrorFunction (NodeRow & _row, int _childNodeCount) : row(_row), childNodeCount(_childNodeCount) {}
        
        /**
         * Copy constructor
         */
        ChildRowEntropyErrorFunction (const ChildRowEntropyErrorFunction & other) : row(other.row), childNodeCount(other.childNodeCount) {}
        
        /**
         * Assignment operator
         */
        ChildRowEntropyErrorFunction & operator=(const ChildRowEntropyErrorFunction & other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                row = other.row;
                childNodeCount = other.childNodeCount;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~ChildRowEntropyErrorFunction() {}
        
        /**
         * Calculates the error if we split. This function expects the local histograms to be already computed.
         */
        float error() const
        {
            float result = 0.;
            
            int classCount = (*row.begin())->getClassHistogram()->size();
            
            // We build up a histogram for every (virtual) child node
            ClassHistogram* histograms = new ClassHistogram[childNodeCount];
            
            // Initialize the histograms
            for (int i = 0; i < childNodeCount; i++)
            {
                histograms[i].resize(classCount);
            }

            // We store the total data count in order to calculate the weighted entropy correctly
            int dataCount = 0;

            // Compute the histograms for all child nodes
            for (NodeRow::iterator it = row.begin(); it != row.end(); ++it)
            {
                int leftNode = (*it)->getTempLeft();
                int rightNode = (*it)->getTempRight();
                ClassHistogram* leftHistogram = (*it)->getLeftHistogram();
                ClassHistogram* rightHistogram = (*it)->getRightHistogram();

                // Add the values to the histogram
                for (int i = 0; i < classCount; i++)
                {
                    histograms[leftNode].add(i, leftHistogram->at(i));
                    histograms[rightNode].add(i, rightHistogram->at(i));
                }

                dataCount += leftHistogram->getMass() + rightHistogram->getMass();
            }

            // Calculate the entropy based on the built up histograms
            for (int i = 0; i < childNodeCount; i++)
            {
                result += histograms[i].getMass()/static_cast<float>(dataCount) * histograms[i].entropy();
            }

            delete[] histograms;

            return result;
        }
    };

    /**
     * Calculates the entropy error based on a child row
     */
    class ThresholdEntropyErrorFunction {
    private:
        /**
         * The row of nodes
         */
        NodeRow & row;
        /**
         * The parent node that we optimize
         */
        TrainingDAGNode::ptr parent;
        /**
         * The left histogram base
         */
        ClassHistogram leftHistogram;
        /**
         * The left histogram base
         */
        ClassHistogram rightHistogram;
        /**
         * Current left/right histogram
         */
        EfficientEntropyHistogram cleftHistogram;
        EfficientEntropyHistogram crightHistogram;
        
    public:
        /**
         * Default constructor
         * @return 
         */
        ThresholdEntropyErrorFunction (NodeRow & _row, TrainingDAGNode::ptr parent) : row(_row), parent(parent) {}
        
        /**
         * Copy constructor
         */
        ThresholdEntropyErrorFunction (const ThresholdEntropyErrorFunction & other) : row(other.row), parent(other.parent) {}
        
        /**
         * Assignment operator
         */
        ThresholdEntropyErrorFunction & operator=(const ThresholdEntropyErrorFunction & other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                row = other.row;
                parent = other.parent;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~ThresholdEntropyErrorFunction() {}
        
        /**
         * Initializes the left/right histogram
         */
        void initHistograms();
        
        /**
         * Resets the current left and right histograms
         */
        void resetHistograms()
        {
            cleftHistogram.reset();
            crightHistogram.reset();
            
            // Initialize the histograms
            for (int i = 0; i < leftHistogram.size(); i++)
            {
                cleftHistogram.set(i, leftHistogram.at(i));
                crightHistogram.set(i, rightHistogram.at(i) + parent->getClassHistogram()->at(i));
            }
            cleftHistogram.initEntropies();
            crightHistogram.initEntropies();
        }
        
        /**
         * Moves one training example from the right to the left histogram
         */
        void move(int classLabel)
        {
            cleftHistogram.addOne(classLabel);
            crightHistogram.subOne(classLabel);
        }
        
        /**
         * Calculates the error if we split. This function expects the local histograms to be already computed.
         */
        float error() const
        {
            return cleftHistogram.entropy() + crightHistogram.entropy();
        }
    };

    /**
     * Calculates the entropy error based on a child row
     */
    class AssignmentEntropyErrorFunction {
    private:
        /**
         * The row of nodes
         */
        NodeRow & row;
        /**
         * The parent node that we optimize
         */
        TrainingDAGNode::ptr parent;
        /**
         * All child node histograms and data counts
         */
        ClassHistogram* histograms;
        float* entropies;
        int dataCount;
        
        /**
         * Number of child nodes
         */
        int childNodeCount;
        
    public:
        /**
         * Default constructor
         * @return 
         */
        AssignmentEntropyErrorFunction (NodeRow & _row, TrainingDAGNode::ptr parent, int childNodeCount) : row(_row), parent(parent), childNodeCount(childNodeCount) {}
        
        /**
         * Copy constructor
         */
        AssignmentEntropyErrorFunction (const AssignmentEntropyErrorFunction & other) : row(other.row), parent(other.parent), childNodeCount(other.childNodeCount) {
        }
        
        /**
         * Assignment operator
         */
        AssignmentEntropyErrorFunction & operator=(const AssignmentEntropyErrorFunction & other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                row = other.row;
                parent = other.parent;
                childNodeCount = other.childNodeCount;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~AssignmentEntropyErrorFunction() {
            delete[] histograms;
            delete[] entropies;
        }
        
        /**
         * Initializes the left/right histogram
         */
        void initHistograms();
        
        /**
         * Calculates the error if we split. This function expects the local histograms to be already computed.
         */
        float error() const 
        {
            float error = 0;

            for (int i = 0; i < childNodeCount; i++)
            {
                if (i == parent->getTempLeft() && i != parent->getTempRight())
                {
                    ClassHistogram* leftHistogram = parent->getLeftHistogram();

                    error += leftHistogram->getMass(histograms[i]) * leftHistogram->entropy(histograms[i]);
                }
                else if (i == parent->getTempRight() && i != parent->getTempLeft())
                {
                    ClassHistogram* rightHistogram = parent->getRightHistogram();

                    error += rightHistogram->getMass(histograms[i]) * rightHistogram->entropy(histograms[i]);
                }
                else if (i == parent->getTempRight() && i == parent->getTempLeft())
                {
                    ClassHistogram* leftHistogram = parent->getLeftHistogram();
                    ClassHistogram* rightHistogram = parent->getRightHistogram();

                    error += rightHistogram->getMass(histograms[i], *leftHistogram) * rightHistogram->entropy(histograms[i], *leftHistogram);
                }
                else
                {
                    error += histograms[i].getMass() * entropies[i];
                }
            }

            return error/dataCount;
        }
    };
    
    
    /**
     * This comparator class allows us to sort a training set according to one feature dimension. 
     */
    class TrainingExampleComparator {
    private:
        /**
         * The feature dimension to check
         */
        int featureDimension;
        
    public:
        /**
         * Default constructor
         * 
         * @param featureDimension
         */
        TrainingExampleComparator(int featureDimension) : featureDimension(featureDimension) {}
        
        /**
         * Copy constructor
         */
        TrainingExampleComparator(const TrainingExampleComparator & other) : featureDimension(other.featureDimension) {}
        
        /**
         * Assignment operator
         */
        TrainingExampleComparator & operator=(const TrainingExampleComparator & other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                featureDimension = other.featureDimension;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~TrainingExampleComparator() {}
        
        /**
         * Compares two training examples
         * 
         * @param lhs
         * @param rhs
         * @return whether or not a[f] < b[f]
         */
        bool operator() (const TrainingExample::ptr lhs, const TrainingExample::ptr rhs)
        {
            return (lhs->getDataPoint()->at(featureDimension) < rhs->getDataPoint()->at(featureDimension));
        }
        
        /**
         * Factory class for TrainingExampleComparator
         */
        class Factory {
        public:
            /**
             * Creates a new instance of TrainingExampleComparator. 
             * Caution: this does not return a reference or a (shared) pointer
             * 
             * @param _featureDimension The feature dimension that shall be sorted
             * @return new comparator instance
             */
            static TrainingExampleComparator create(int _featureDimension)
            {
                return TrainingExampleComparator(_featureDimension);
            }
        };
    };
    
    /**
     * This is a collection of training utility functions used during training that could not be bound to single 
     * objects because they e.g. were derived from some std class and we didn't want to cope with all the stress of
     * defining appropriate constructors/destructor etc. 
     */
    class TrainingUtil {
    public:
        /**
         * Computes a histogram from a training set. The histogram must already be set up correctly (e.g. the number
         * of bins must equal the number of classes). 
         * 
         * @param _hist The histogram
         * @param _trainingSet The training set
         */
        static void computHistogram(ClassHistogram & _hist, TrainingSet::ptr _trainingSet)
        {
            // Initialize the histogram
            _hist.reset();
            
            // Compute the histogram
            for (TrainingSet::iterator i = _trainingSet->begin(); i != _trainingSet->end(); ++i)
            {
                _hist.add((*i)->getClassLabel(), 1);
            }
        }
        
        /**
         * Computes and returns the arg max over a class histogram
         * 
         * @param _hist The class histogram
         * @return The most probable class label
         */
        static ClassLabel histogramArgMax(const ClassHistogram & _hist)
        {
            int classCount = _hist.size();
            ClassLabel bestClassLabel = -1;
            int bestScore = 0;
            
            // Initialize the histogram
            for (int i = 0; i < classCount; i++)
            {
                if (_hist.at(i) > bestScore)
                {
                    bestClassLabel = i;
                    bestScore = _hist.at(i);
                }
            }
            
            return bestClassLabel;
        }
        
        /**
         * Computes and returns the max over a class histogram
         * 
         * @param _hist The class histogram
         * @return The mode of the distribution
         */
        static int histogramMax(const ClassHistogram & _hist)
        {
            int bestScore = 0;
            
            // Initialize the histogram
            for (int i = _hist.begin(); i != _hist.end(); ++i)
            {
                if (_hist.at(i) > bestScore)
                {
                    bestScore = _hist.at(i);
                }
            }
            
            return bestScore;
        }
        
        /**
         * Returns true if the histogram is concentrated on one single class
         * 
         * @param _hist The class histogram
         * @return true if the histogram is concentrated on one class
         */
        static bool histogramIsDirichlet(const ClassHistogram & _hist)
        {
            return TrainingUtil::histogramIsAlmostDirichlet(_hist, 0);
        }
        
        /**
         * Returns true if the histogram is almost concentrated on one single class
         * 
         * @param _hist The class histogram
         * @param _threshold The level of noise that is allowed in the non mode bins
         * @return true if the histogram is concentrated on one class
         */
        static bool histogramIsAlmostDirichlet(const ClassHistogram & _hist, int _threshold)
        {
            bool foundPeak = false;
            
            // Initialize the histogram
            for (int i = _hist.begin(); i != _hist.end(); ++i)
            {
                if (_hist.at(i) > _threshold)
                {
                    // If this is the first peak, there is no problem
                    if (!foundPeak)
                    {
                        foundPeak = true;
                    }
                    else
                    {
                        // This is the second peak, this is no dirichlet distribution
                        return false;
                    }
                }
            }
            
            return true;
        }
    };
    
    /**
     * Calculates some statistics over a trained jungle
     */
    class TrainingStatistics : public Statistics {
    public:
        typedef TrainingStatistics self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Calculates the error on a training set
         * 
         * @param _jungle
         * @param _trainingSet
         * @return Training error
         */
        float trainingError(Jungle::ptr _jungle, TrainingSet::ptr _trainingSet);
        
        /**
         * Calculates a confusion matrix on a training set
         * 
         * @param _jungle
         * @param _trainingSet
         * @return the confusion matrix
         */
        // FIXME
        // Matrix confusionMatrix(Jungle::ptr _jungle, TrainingSet::ptr _trainingSet);
        
        /**
         * A factory for this class
         */
        class Factory {
        public:
            /**
             * Creates a new blank instance
             * 
             * @return blank instance
             */
            static TrainingStatistics::ptr create() 
            {
                return TrainingStatistics::ptr(new self());
            }
        };
    };
}

#endif
