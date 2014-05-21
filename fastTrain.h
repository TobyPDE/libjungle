/* 
 * File:   fastTrain.h
 * Author: Toby
 *
 * Created on May 19, 2014, 11:09 AM
 */

#ifndef FASTTRAIN_H
#define FASTTRAIN_H

#include "jungleTrain.h"

namespace decision_jungle {
    typedef unsigned int FastNode;
    
    typedef double* FastDataPoint;
    typedef struct {
        ClassLabel label;
        FastDataPoint* featureVector;
    } FastTrainingExample;
    typedef std::vector<FastTrainingExample*> FastTrainingSet;
    
    /**
     * This class is used for fast learning and prediction of DAGs.
     * Each node is represented by an index. The values (threshold, feature IDs, 
     * child assignments) are stored in arrays respectively.
     */
    class FastDAG : public AbstractTrainer {
    private:
        /**
         * The total number of nodes in this DAG
         */
        FastNode nodeCount;
        /**
         * The thresholds for all the nodes
         */
        double* thresholds;
        /**
         * The feature IDs for all nodes
         */
        int* featureIDs;
        /**
         * The left child nodes for each node
         */
        FastNode* leftChildren;
        /**
         * The right child nodes for each node
         */
        FastNode* rightChildren;
        /**
         * The predicted label for each node. Internal nodes don't get any label.
         * They always predict class 0.
         */
        ClassLabel* labels;
        /**
         * The histograms at the childnodes
         * FIXME
         */
        
        /**********************************************************************
         * The attributes below are used exclusively for training.
         *********************************************************************/
        
        /**
         * The feature dimension
         */
        int _featureDim;
        /**
         * The total number of classes
         */
        int _classCount;
        /**
         * The total number of training examples
         */
        int _trainingSetSize;
        /**
         * The current collection of parent nodes
         */
        FastNode* _parents;
        /**
         * The current collection of child nodes
         */
        FastNode* _children;
        /**
         * The class histograms for the virtual child nodes
         */
        float* _virtualHistograms;
        /**
         * The class histograms at the childnodes
         */
        float* _childHistograms;
        /**
         * The entropies of the virtual child histograms
         */
        float* _virtualEntropies;
        /**
         * The entropies of the child node histograms
         */
        float* _childEntropies;
        /**
         * The training sets at the parent nodes
         */
        FastTrainingSet* _parentTrainingSets;
        
        /**
         * Creates a new node in the DAG and returns it's index
         */
        FastNode createNode();
        
    public:
        /**
         * Trains the DAG on a training set
         */
        void train(FastTrainingSet* _trainingSet);
        /**
         * Predicts the class label of a data point
         */
        ClassLabel predict(FastDataPoint* _dataPoint);
    };
    
}

#endif	/* FASTTRAIN_H */

