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
        typedef std::shared_ptr<self> ptr;
        
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
                ptr featureVector (new self(_dim));
                // Initialize the vector
                for (int i = 0; i < _dim; i++)
                {
                    (*featureVector)[i] = 0;
                }
                
                return featureVector;
            }
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
        typedef std::shared_ptr<self> ptr;
        
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
                ClassHistogram::ptr histogram(new ClassHistogram::self(bins));

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
        float confidence;
        
    public:
        typedef PredictionResult self;
        typedef std::shared_ptr<self> ptr;
        
        /**
         * Default constructor
         */
        PredictionResult(ClassLabel _classLabel, float _confidence) : classLabel(_classLabel), confidence(_confidence) {}
        
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
        float getConfidence()
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
            static PredictionResult::ptr create(const ClassLabel _classLabel, const float _confidence)
            {
                PredictionResult::ptr result(new PredictionResult::self(_classLabel, _confidence));
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
                PredictionResult::ptr result(new PredictionResult::self(_classLabel, 0));
                return result;
            }
        };
    };

    /**
     * This class represents a single node in a decision DAG
     */
    class DAGNode {
    public:
        DEFINE_CLASS_PTR(DAGNode)
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
                DAGNode::ptr node(new DAGNode());

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
                return ptr(new Jungle);
            }
        };
    };}
#endif