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
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include "misc.h"
#include "fastlog.h"
#include <cmath>
#define ENTROPY(p) -(p) * fl2(p)

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
        typedef std::vector<float> self;
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
    
    static inline float fl2(float x)
    {
            union { float f; uint32_t i; } vx = { x };
            union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
            float y = float(vx.i); y *= 1.1920928955078125e-7f;
            return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
    }

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
    private:
        /**
         * The number of classes in this histogram
         */
        int bins;
        
        /**
         * The actual histogram
         */
        int* histogram;
        
        /**
         * The integral over the entire histogram
         */
        int mass;
        
    public:
        /**
         * Default constructor
         */
        ClassHistogram() : bins(0), histogram(0), mass(0) { }
        ClassHistogram(int _classCount) : bins(_classCount), histogram(0), mass(0){ resize(_classCount); }
        
        /**
         * Copy constructor
         */
        ClassHistogram(const ClassHistogram & other) 
        {
            resize (other.bins);
            for (int i = 0; i < bins; i++)
            {
                set(i, other.at(i));
            }
            mass = other.mass;
        }
        
        /**
         * Assignment operator
         */
        ClassHistogram & operator= (const ClassHistogram &other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                resize (other.bins);
                for (int i = 0; i < bins; i++)
                {
                    set(i, other.at(i));
                }
                mass = other.mass;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        ~ClassHistogram()
        {
            if (histogram != 0)
            {
                delete[] histogram;
            }
        }
        
        /**
         * Resizes the histogram to a certain size
         */
        void resize(int _classCount)
        {
            // Release the current histogram
            if (histogram != 0)
            {
                delete[] histogram;
                histogram = 0;
            }
            
            // Only allocate a new histogram, if there is more than one class
            if (_classCount > 0)
            {
                histogram = new int[_classCount];
                bins = _classCount;
                
                // Initialize the histogram
                for (int i = 0; i < bins; i++)
                {
                    histogram[i] = 0;
                }
            }
        }
        
        /**
         * Returns the size of the histogram (= class count)
         */
        int size() const { return bins; }
        
        /**
         * Returns the value of the histogram at a certain position. Caution: For performance reasons, we don't
         * perform any parameter check!
         */
        int at(int i) const { return histogram[i]; }
        int get(int i) const { return histogram[i]; }
        void set(int i, int v) { mass -= histogram[i]; mass += v; histogram[i] = v; }
        void add(int i, int v) { mass += v; histogram[i] += v; }
        void sub(int i, int v) { mass -= v; histogram[i] -= v; }
        void addOne(int i) { mass++; histogram[i]++; }
        void subOne(int i) { mass--; histogram[i]--; }
        
        /**
         * Returns the mass
         */
        float getMass() const { return mass; }
        
        /**
         * Returns the mass of the combined histogram (this + _hist)
         */
        float getMass(const ClassHistogram & _hist) const { return getMass() + _hist.getMass(); }
        
        /**
         * Returns the mass of the combined histogram (this + _hist1 + _hist2)
         */
        float getMass(const ClassHistogram & _hist1, const ClassHistogram & _hist2) const { return getMass() + _hist1.getMass() + _hist2.getMass(); }
        
        /**
         * Iterator interface from STL
         */
        int begin() const { return 0; } 
        int end() const { return bins; }
        
        /**
         * Calculates the entropy of a histogram
         * 
         * @return The calculated entropy
         */
        float entropy() const
        {
            float sum = getMass();
            if (sum < 1) return 0;
            
            float entropy = 0;
            
            for (int i = 0; i < bins; i++)
            {
                // Empty bins do not contribute anything
                if (at(i) > 0)
                {
                    entropy += ENTROPY(at(i)/sum);
                }
            }

            return entropy;
        }
        
        /**
         * Calculates the entropy of the histogram (this + _hist)
         * 
         * @return The calculated entropy
         */
        float entropy(const ClassHistogram & _hist) const
        {
            float sum = getMass(_hist);
            
            if (sum < 1) return 0;
            
            float entropy = 0;
            float numerator = 0;
            
            for (int i = 0; i < bins; i++)
            {
                // Empty bins do not contribute anything
                numerator = at(i) + _hist.at(i);
                if (numerator > 0)
                {
                    entropy += ENTROPY(numerator/sum);
                }
            }
            
            return entropy;
        }
        
        /**
         * Calculates the entropy of the histogram (this + _hist)
         * 
         * @return The calculated entropy
         */
        float entropy(const ClassHistogram & _hist1, const ClassHistogram & _hist2) const
        {
            float sum = getMass(_hist1, _hist2);
            
            if (sum < 1) return 0;
            
            float entropy = 0;
            float numerator = 0;
            
            for (int i = 0; i < bins; i++)
            {
                // Empty bins do not contribute anything
                numerator = at(i) + _hist2.at(i) + _hist1.at(i);
                if (numerator > 0)
                {
                    entropy += ENTROPY(numerator/sum);
                }
            }
            
            return entropy;
        }
        
        /**
         * Sets all entries in the histogram to 0
         */
        void reset()
        {
            // Only reset the histogram if there are more than 0 bins
            if (histogram == 0) return;
            
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
            }
            mass = 0;
        }
    };
    
    /**
     * A histogram over the class labels
     */
    class EfficientEntropyHistogram {
    private:
        /**
         * The number of classes in this histogram
         */
        int bins;
        
        /**
         * The actual histogram
         */
        int* histogram;
        
        /**
         * The integral over the entire histogram
         */
        int mass;
        
        /**
         * The entropies for the single bins
         */
        float* entropies;
        
        /**
         * The total entropy
         */
        float totalEntropy;
        
    public:
        /**
         * Default constructor
         */
        EfficientEntropyHistogram() : bins(0), histogram(0), mass(0), entropies(0), totalEntropy(0) { }
        EfficientEntropyHistogram(int _classCount) : bins(_classCount), histogram(0), mass(0), entropies(0), totalEntropy(0) { resize(_classCount); }
        
        /**
         * Copy constructor
         */
        EfficientEntropyHistogram(const EfficientEntropyHistogram & other) 
        {
            resize (other.bins);
            for (int i = 0; i < bins; i++)
            {
                set(i, other.at(i));
            }
            mass = other.mass;
        }
        
        /**
         * Assignment operator
         */
        EfficientEntropyHistogram & operator= (const EfficientEntropyHistogram &other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                resize (other.bins);
                for (int i = 0; i < bins; i++)
                {
                    set(i, other.at(i));
                }
                mass = other.mass;
            }
            return *this;
        }
        
        /**
         * Destructor
         */
        ~EfficientEntropyHistogram()
        {
            if (histogram != 0)
            {
                delete[] histogram;
            }
            if (entropies != 0)
            {
                delete[] entropies;
            }
        }
        
        /**
         * Resizes the histogram to a certain size
         */
        void resize(int _classCount)
        {
            // Release the current histogram
            if (histogram != 0)
            {
                delete[] histogram;
                histogram = 0;
            }
            if (entropies != 0)
            {
                delete[] entropies;
                entropies = 0;
            }
            
            // Only allocate a new histogram, if there is more than one class
            if (_classCount > 0)
            {
                histogram = new int[_classCount];
                entropies = new float[_classCount];
                bins = _classCount;
                
                // Initialize the histogram
                for (int i = 0; i < bins; i++)
                {
                    histogram[i] = 0;
                    entropies[i] = 0;
                }
            }
        }
        
        /**
         * Returns the size of the histogram (= class count)
         */
        int size() const { return bins; }
        
        /**
         * Returns the value of the histogram at a certain position. Caution: For performance reasons, we don't
         * perform any parameter check!
         */
        int at(int i) const { return histogram[i]; }
        int get(int i) const { return histogram[i]; }
        void set(int i, int v) { mass -= histogram[i]; mass += v; histogram[i] = v; }
        void add(int i, int v) { mass += v; histogram[i] += v; }
        void sub(int i, int v) { mass -= v; histogram[i] -= v; }
        void addOne(int i)
        {
            totalEntropy += ENTROPY(getMass());
            mass++;
            totalEntropy += -ENTROPY(getMass());
            histogram[i]++;
            totalEntropy -= entropies[i];
            entropies[i] = ENTROPY(histogram[i]); 
            totalEntropy += entropies[i];
        }
        void subOne(int i)
        { 
            totalEntropy += ENTROPY(getMass());
            mass--; 
            totalEntropy += -ENTROPY(getMass());
            
            histogram[i]--;
            totalEntropy -= entropies[i];
            if (histogram[i] < 1)
            {
                entropies[i] = 0;
            }
            else
            {
                entropies[i] = ENTROPY(histogram[i]); 
            }
            totalEntropy += entropies[i];
        }
        
        /**
         * Returns the mass
         */
        float getMass() const { return mass; }
        
        /**
         * Calculates the entropy of a histogram
         * 
         * @return The calculated entropy
         */
        float entropy() const
        {
            return totalEntropy;
        }
        
        /**
         * Initializes all entropies
         */
        void initEntropies()
        {
            if (getMass() < 1)
            {
                totalEntropy = 0;
                for (int i = 0; i < bins; i++)
                {
                    entropies[i] = 0;
                }
            }
            else
            {
                totalEntropy = -ENTROPY(getMass());
                for (int i = 0; i < bins; i++)
                {
                    if (at(i) == 0) continue;

                    entropies[i] = ENTROPY(at(i));
                    
                    totalEntropy += entropies[i];
                }
            }
        }
        
        /**
         * Sets all entries in the histogram to 0
         */
        void reset()
        {
            // Only reset the histogram if there are more than 0 bins
            if (histogram == 0) return;
            
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
                entropies[i] = 0;
            }
            totalEntropy = 0;
            mass = 0;
        }
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
        float threshold;
        
        /**
         * The left child node
         */
        ptr left;
        
        /**
         * The right child node
         */
        ptr right;
        
        /**
         * Temporary left node assignment. Used during training and model loading
         */
        int tempLeft;
        
        /**
         * Temporary right node assignment. Used during training and model loading
         */
        int tempRight;
        
        /**
         * Assigned class label for this node
         */
        ClassLabel classLabel;
        
        /**
         * Class histogram for this node
         */
        ClassHistogram classHistogram;
        
        /**
         * The ID of this node. This is used in order to serialize nodes
         */
        int ID;
        
        /**
         * Initializes all parameters
         */
        void initParameters();
        
    public:
        virtual ~DAGNode() {}
        
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
            DAGNode::ptr current = 0;
            
            while (queue.size() > 0)
            {
                current = queue.back();
                queue.pop_back();
                
                if (deletionSet.find(current) != deletionSet.end()) continue;
                
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
        float getThreshold() const
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
        void setThreshold(float _threshold)
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
        ClassHistogram* getClassHistogram()
        {
            return &classHistogram;
        }
        
        /**
         * Returns the class histogram
         * 
         * @return Class histogram
         */
        const ClassHistogram* getClassHistogram() const
        {
            return &classHistogram;
        }
    
        /**
         * Returns the node ID
         */
        int getID() const
        {
            return ID;
        }
        
        /**
         * Sets the node ID
         */
        void setID(int _ID)
        {
            ID = _ID;
        }
                
        /**
         * Sets the temporary left assignment
         * 
         * @param _tempLeft New assignment
         */
        void setTempLeft(int _tempLeft)
        {
            tempLeft = _tempLeft;
        }
        
        /**
         * Returns the temporary left assignment
         * 
         * @return temporary left assignment
         */
        int getTempLeft()
        {
            return tempLeft;
        }
        
        /**
         * Sets the temporary right assignment
         * 
         * @param _tempRight New assignment
         */
        void setTempRight(int _tempRight)
        {
            tempRight = _tempRight;
        }
        
        /**
         * Returns the temporary right assignment
         * 
         * @return temporary right assignment
         */
        int getTempRight()
        {
            return tempRight;
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
            static void init(DAGNode::ptr node, int classCount)
            {
                node->setFeatureID(0);
                node->setThreshold(0);
                node->setClassLabel(0);
                node->setLeft(0);
                node->setRight(0);
            }
            
        public:
            /**
             * Creates a new initialized DAG node (class label = 0, feature dimension = 0)
             * 
             * @return initialized DAG node
             */
            static DAGNode::ptr create(int classCount)
            {
                DAGNode::ptr node = new DAGNode();

                // Initialize the node
                Factory::init(node, classCount);
                
                return node;
            }
            
            /**
             * Serializes a node in order to save it to a file
             */
            static void serialize(DAGNode::ptr node, bool isRoot, std::ofstream & outfile);
            
            /**
             * Unserializes a node from a model file
             */
            static DAGNode::ptr unserialize(const std::vector<std::string> & row);
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
            
            /**
             * Serializes the complete jungle in order to save it to a file
             */
            static void serialize(Jungle::ptr jungle, const std::string & filename)
            {
                // Open the file
                std::ofstream outfile(filename);
                
                // Did we open the file successfully?
                if (!outfile.is_open())
                {
                    throw RuntimeException("Could not open model file.");
                    return;
                }
                
                // Give all nodes an ID
                int ID = 1;
                for (std::set<DAGNode::ptr>::iterator it = jungle->dags.begin(); it != jungle->dags.end(); ++it)
                {
                    // Put all nodes into a set and then iterate over the set to delete all nodes
                    std::vector<DAGNode::ptr> queue;
                    std::set<DAGNode::ptr> nodeSet;

                    // Start with the root node
                    queue.push_back(*it);
                    DAGNode::ptr current = 0;

                    while (queue.size() > 0)
                    {
                        current = queue.back();
                        queue.pop_back();

                        if (nodeSet.find(current) != nodeSet.end()) continue;

                        nodeSet.insert(current);
                        current->setID(ID++);

                        if (current->getLeft() != 0)
                        {
                            queue.push_back(current->getLeft());
                            queue.push_back(current->getRight());
                        }
                    }
                    
                    // Save all the nodes
                    for (std::set<DAGNode::ptr>::iterator nodeIt = nodeSet.begin(); nodeIt != nodeSet.end(); ++nodeIt)
                    {
                        DAGNode::Factory::serialize(*nodeIt, *nodeIt == *it, outfile);
                    }
                }
                
                outfile.close();
            }
            
            /**
             * Loads a jungle from a model file
             */
            static Jungle::ptr createFromFile(const std::string & filename, bool);
        };
        friend class Jungle::Factory;
    };
    
    /**
     * This class calculates some statistics. 
     */
    class Statistics {
    public:
        typedef Statistics self;
        typedef std::shared_ptr<self> ptr;
        
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
            float progress = 0;
            // Relative progress
            if (total <= 0)
            {
                progress = 1.;
            }
            else
            {
                progress = _state/static_cast<float>(total);
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