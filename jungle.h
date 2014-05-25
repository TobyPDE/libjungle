/**
 * This file contains the basic definitions for all the functionality that is needed in order to train a jungle/DAG, 
 * predict new class labels/structures, save and load trained models. 
 * 
 * @author Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
 * @version 1.0
 */
#ifndef JUNGLE_H
#define JUNGLE_H

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include "misc.h"

namespace decision_jungle
{
    /**
     * This exception is thrown when something unexpected happens during execution. 
     */
    DEFINE_EXCEPTION(RuntimeException)
    
    /**
     * This exception is thrown when some configuration parameters are invalid
     */
    DEFINE_EXCEPTION(ConfigurationException)
    
    /**
     * Forward declarations
     */
    class DAGNode;
    class PredictionResult;
    class ClassHistogram;
    class DAGNode;
    
    /**
     * At the moment, we only consider integer class labels
     */
    typedef int ClassLabel;
    
    /**
     * A data point is thought to be a feature vector
     */
    class DataPoint {
    public:
        typedef std::vector<double> self;
        typedef self* ptr;
        
        /**
         * A factory for data points
         */
        class Factory {
        public:
            /**
             * Creates a new zero initialized feature vector of dimension _dim
             * 
             * @param _dim The feature dimension
             * @return Initialized feature vector
             */
            static ptr createZeroInitialized(int _dim)
            {
                // The size must not be 0, then the vector would have dimension zero
                if(_dim <= 0)
                {
                    throw RuntimeException("Invalid vector dimension.");
                }
                
                ptr featureVector = new self(_dim);
                
                // Initialize the vector
                for (int i = 0; i < _dim; i++)
                {
                    (*featureVector)[i] = 0;
                }
                
                return featureVector;
            }
            
            /**
             * Creates a new data point from a row from a data set
             * 
             * @param _row The vector of strings representing the training example
             * @return The created training example
             */
            static ptr createFromFileRow(const std::vector<std::string> & _row);
        };
    };
    
    
    /**
     * A data set is a collection of data points
     */
    class DataSet {
    public:
        // We use a vector because we need to sort the set 
        typedef std::vector<DataPoint::ptr> self;
        typedef self::iterator iterator;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * A factory class for training sets
         */
        class Factory {
        public:
            /**
             * Creates a new blank data set
             */
            static DataSet::ptr create() 
            {
                DataSet::ptr result (new self());
                return result;
            }
            
            /**
             * Loads a training set from a file
             * 
             * @param _fileName The filename
             * @param _verboseMode
             * @return The loaded training set
             */
            static DataSet::ptr createFromFile(const std::string & _fileName, bool _verboseMode);
        };
    };
    
    /**
     * A histogram over the class labels
     */
    class ClassHistogram {
    public:
        typedef int value;
        typedef std::vector<value> self;
        typedef self::iterator iterator;
        typedef self* ptr;
        
        /**
         * A Factory for class histograms
         */
        class Factory {
        public:
            /**
             * Creates a new initialized class histogram with a fixed number of bins
             * 
             * @param bins The number of histogram bins
             * @return pointer to the histogram
             */
            static ClassHistogram::ptr createEmpty(int bins) 
            {
                ClassHistogram::ptr histogram = new ClassHistogram::self(bins);

                // Initialize all bins with zero
                for (int i = 0; i < bins; i++)
                {
                    (*histogram)[i] = 0;
                }

                return histogram;
            }
            
            /**
             * Clones a histogram
             * 
             * @param _hist The histogram to clone
             * @return cloned histogram
             */
            static ClassHistogram::ptr clone(ClassHistogram::ptr _hist)
            {
                ClassHistogram::ptr histogram(new ClassHistogram::self(*_hist));
                return histogram;
            }
        };
    };
    
    /**
     * Classification result: It consists of the predicted class label and the confidence
     */
    class PredictionResult {
    private:
        /**
         * The chosen class label
         */
        ClassLabel classLabel;
        
        /**
         * The prediction confidence
         */
        double confidence;
        
    public:
        typedef PredictionResult self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Default constructor
         */
        PredictionResult(ClassLabel _classLabel, double _confidence) : classLabel(_classLabel), confidence(_confidence) {}
        
        /**
         * Copy constructor
         */
        PredictionResult(const PredictionResult &copy) : classLabel(copy.classLabel), confidence(copy.confidence) {}
        
        /**
         * Assignment operator
         */
        PredictionResult & operator= (const PredictionResult &other)
        {
            // Do not make self copies
            if (this != &other)
            {
                classLabel = other.classLabel;
                confidence = other.confidence;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        virtual ~PredictionResult() {}
        
        /**
         * Returns the predicted class label
         * 
         * @return class label
         */
        ClassLabel getClassLabel()
        {
            return classLabel;
        }
        
        /**
         * Returns the prediction confidence
         * 
         * @return prediction confidence
         */
        double getConfidence()
        {
            return confidence;
        }
        
        /**
         * A factory for the prediction results
         */
        class Factory {
        public:
            /**
             * Creates a new prediction result from a class label and a confidence value
             * 
             * @param _classLabel The predicted class label
             * @param _confidence The confidence
             * @return new Prediction result
             */
            static PredictionResult::ptr create(const ClassLabel _classLabel, const double _confidence)
            {
                PredictionResult::ptr result (new PredictionResult::self(_classLabel, _confidence));
                return result;
            }

            /**
             * Creates a new prediction result from a class label. The confidence is set to zero.
             * 
             * @param _classLabel The predicted class label
             * @return new Prediction result
             */
            static PredictionResult::ptr create(const ClassLabel _classLabel)
            {
                PredictionResult::ptr result (new PredictionResult::self(_classLabel, 0));
                return result;
            }
        };
    };

    /**
     * This class represents a single node in a decision DAG
     */
    class DAGNode {
    public:
        typedef DAGNode self;
        typedef self* ptr;
        
    private:
        /**
         * The feature ID (feature vector index to test) that is tested at this node
         */
        int featureID;
        
        /**
         * The applied threshold
         */
        double threshold;
        
        /**
         * The left child node
         */
        ptr left;
        
        /**
         * The right child node
         */
        ptr right;
        
        /**
         * Assigned class label for this node
         */
        ClassLabel classLabel;
        
        /**
         * Class histogram for this node
         */
        ClassHistogram::ptr classHistogram;
        
        /**
         * Initializes all parameters
         */
        void initParameters();
        
    public:
        virtual ~DAGNode()
        {
            delete classHistogram;
        }
        
        /**
         * Deletes a DAG given by its root node
         */
        static void deleteDAG(DAGNode::ptr root)
        {
            // Put all nodes into a set and then iterate over the set to delete all nodes
            std::vector<DAGNode::ptr> queue;
            std::set<DAGNode::ptr> deletionSet;
            
            // Start with the root node
            queue.push_back(root);
            
            while (queue.size() > 0)
            {
                DAGNode::ptr current = queue.back();
                queue.pop_back();
                
                deletionSet.insert(current);
                
                if (current->getLeft() != 0)
                {
                    queue.push_back(current->getLeft());
                    queue.push_back(current->getRight());
                }
            }
            
            for (std::set<DAGNode::ptr>::iterator it = deletionSet.begin(); it != deletionSet.end(); ++it)
            {
                delete *it;
            }
        }
        
        /**
         * Returns the feature ID
         * 
         * @return selected feature ID
         */
        int getFeatureID() const
        {
            return featureID;
        }
        
        /**
         * Returns the threshold value
         * 
         * @return selected threshold value
         */
        double getThreshold() const
        {
            return threshold;
        }
        
        /**
         * Returns the left child node
         * 
         * @return left node or null if there is no left child node
         */
        ptr getLeft() const
        {
            return left;
        }
        
        /**
         * Returns the right child node
         * 
         * @return right node or null if there is no right child node
         */
        ptr getRight() const
        {
            return right;
        }
        
        /**
         * Sets the feature ID
         * 
         * @param _featureID
         */
        void setFeatureID(int _featureID)
        {
            featureID = _featureID;
        }
        
        /**
         * Sets the threshold
         * 
         * @param _threshold The new threshold
         */
        void setThreshold(double _threshold)
        {
            threshold = _threshold;
        }
        
        /**
         * Sets the left child node
         * 
         * @param _left The new left child node
         */
        void setLeft(ptr _left)
        {
            left = _left;
        }
        
        /**
         * Sets the right child node
         * 
         * @param _right The new right child node
         */
        void setRight(ptr _right)
        {
            right = _right;
        }

        /**
         * Sets the class label
         * 
         * @param _classLabel The new class label
         */
        void setClassLabel(ClassLabel _classLabel)
        {
            classLabel = _classLabel;
        }
        
        /**
         * Sets the class histogram
         * 
         * @param _classHistogram The class histogram
         */
        void setClassHistogram(ClassHistogram::ptr _classHistogram)
        {
            classHistogram = _classHistogram;
        }
        
        /**
         * Classifies a new data point given by a feature vector
         * 
         * @return Classification result (class label and confidence)
         */
        PredictionResult::ptr predict(DataPoint::ptr featureVector) const;
        
        /**
         * Returns the class label
         * 
         * @return the class label for this node
         */
        ClassLabel getClassLabel() const
        {
            return classLabel;
        }
        
        /**
         * Returns the class histogram
         * 
         * @return Class histogram
         */
        ClassHistogram::ptr getClassHistogram() const
        {
            return classHistogram;
        }
    
        /**
         * This is only for debug purposes
         */
        void traverse()
        {
            printf("%p: [f: %d, t: %2.5f, l: %p, r: %p] -> %d\n", this, getFeatureID(), getThreshold(), getLeft(), getRight(), getClassLabel());
            if (left)
            {
                left->traverse();
            }
            if (right)
            {
                right->traverse();
            }
        }
        
        /**
         * A factory for DAGNodes
         */
        class Factory {
        protected:
            /**
             * Initializes a note
             */
            static void init(DAGNode::ptr node)
            {
                node->setFeatureID(0);
                node->setThreshold(0);
                node->setClassLabel(0);
                node->setLeft(0);
                node->setRight(0);
                node->setClassHistogram(ClassHistogram::Factory::createEmpty(0));
            }
            
        public:
            /**
             * Creates a new initialized DAG node (class label = 0, feature dimension = 0)
             * 
             * @return initialized DAG node
             */
            static DAGNode::ptr create()
            {
                DAGNode::ptr node = new DAGNode();

                // Initialize the node
                Factory::init(node);
                
                return node;
            }
        };        
    };
    
    /**
     * A decision jungle is a collection is decision DAGs. 
     */
    class Jungle {
    private:
        /**
         * The trained DAG root nodes
         */
        std::set<DAGNode::ptr> dags;
        
    public:
        typedef Jungle self;
        typedef std::shared_ptr<self> ptr;
        
        virtual ~Jungle()
        {
            // Delete all DAGs
            for (std::set<DAGNode::ptr>::iterator it = dags.begin(); it != dags.end(); ++it)
            {
                DAGNode::deleteDAG(*it);
            }
        }
        
        /**
         * Returns the trained DAGs
         * 
         * @return List of trained DAGs
         */
        std::set<DAGNode::ptr> & getDAGs()
        {
            return dags;
        }
        
        /**
         * Classifies a new data point given by a feature vector
         * 
         * @return Classification result (class label and confidence)
         */
        PredictionResult::ptr predict(DataPoint::ptr featureVector) const;
        
        /**
         * Factory for decision jungles
         */
        class Factory {
        public:
            /**
             * Creates a new blank decision jungle
             * 
             * @return blank decision jungle
             */
            static Jungle::ptr create()
            {
                return Jungle::ptr(new Jungle);
            }
        };
    };
    
    /**
     * This class calculates some statistics. 
     */
    class Statistics {
    public:
        typedef Statistics self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Calculates a histogram of predicted class labels for a data set
         * 
         * @param _jungle the learned jungle
         * @param _dataSet The dataset to classify
         * @return A class histogram
         */
        ClassHistogram::ptr predictionHistogram(Jungle::ptr _jungle, DataSet::ptr _dataSet);
        
        /**
         * A factory for this class
         */
        class Factory {
        public:
            /**
             * Creates a new blank statistics instance
             * 
             * @return New instance
             */
            static Statistics::ptr create() 
            {
                return Statistics::ptr(new self());
            }
        };
    };
    
    /**
     * This class displays a progress bar in the command line
     */
    class ProgressBar {
    private:
        /**
         * The width or the progress bar
         */
        int width;
        /**
         * The current status of the bar
         */
        int state;
        /**
         * The total number of elements
         */
        int total;
        /**
         * Last upper bound. Don't repaint, if the bar didn't change
         */
        int _lastUpperBound;

    public:
        typedef ProgressBar self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Default constructor
         */
        ProgressBar(int _width, int _total) : width(_width), state(0), total(_total), _lastUpperBound(0) {}
        /**
         * Copy constructor
         */
        ProgressBar(const ProgressBar & other) : width(other.width), state(other.state), total(other.total), _lastUpperBound(0) {}
        /**
         * Destructor
         */
        ~ProgressBar() {}
        /**
         * Assignment operator
         */
        ProgressBar & operator= (const ProgressBar &other)
        {
            // Do not make self copies
            if (this != &other)
            {
                width = other.width;
                state = other.state;
                total = other.total;
                _lastUpperBound = 0;
            }
            return *this;
        }
        
        /**
         * Prints the progress bar with updated state
         * 
         * @param _state The new state
         */
        void update(int _state)
        {
            state = _state;
            double progress = 0;
            // Relative progress
            if (total <= 0)
            {
                progress = 1.;
            }
            else
            {
                progress = _state/static_cast<double>(total);
            }
                
            if (static_cast<int>(progress*width) == _lastUpperBound && _lastUpperBound > 0)
            {
                return;
            }

            _lastUpperBound = static_cast<int>(progress*width);

            printf("\r[");
            for (int j = 0; j < width; j++)
            {
                if (j <= progress*width)
                {
                    printf("*");
                }
                else
                {
                    printf(" ");
                }
            }
            printf("] %4d/%4d (%2.1f%%)", _state, total, progress*100);
            
            // Stop when we reached the end
            if (state >= total)
            {
                printf("\n");
            }

            std::cout.flush();
        }
        
        
        /**
         * Prints the progress bar with increased state
         */
        void update()
        {
            update(state + 1);
        }

        /**
         * A factory method for the progress bar
         */
        class Factory {
        public:
            /**
             * Creates a new progress bar
             * 
             * @param _width
             * @param _total
             * @return new progress bar
             */
            static ProgressBar::ptr create(int _width, int _total)
            {
                return ptr(new self(_width, _total));
            }
            
            /**
             * Creates a new progress bar with default parameters
             * 
             * @param _total
             * @return new progress bar
             */
            static ProgressBar::ptr create(int _total)
            {
                return ProgressBar::Factory::create(50, _total);
            }
        };
    };
}
#endif