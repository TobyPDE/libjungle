#include <map>
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include "jungle.h"

using namespace JunglePP;

PredictionResult::ptr DAGNode::predict(DataPoint::ptr featureVector) const
{
    // Get the leaf node
    const DAGNode* leafNode = getLeafNode(featureVector);
    
    // Compute the relative confidence
    if (leafNode->getClassHistogram()->getMass() > 0)
    {
        return PredictionResult::Factory::create(leafNode->getClassLabel(), leafNode->getClassHistogram()->at(leafNode->getClassLabel())/leafNode->getClassHistogram()->getMass());
    }
    else
    {
        return PredictionResult::Factory::create(leafNode->getClassLabel(), 0);
    }
}

const DAGNode* DAGNode::getLeafNode(DataPoint::ptr featureVector) const
{
    // Does this node have child nodes?
    if (!left)
    {
        // Nope, this is a leaf node
        return this;
    }
    else
    {
        if (featureVector->at(featureID) <= threshold)
        {
            return left->getLeafNode(featureVector);
        }
        else
        {
            return right->getLeafNode(featureVector);
        }
    }
}

PredictionResult::ptr Jungle::predict(DataPoint::ptr featureVector) const
{
    // Use a majority vote
    std::map<ClassLabel, float> votes;
    
    for (std::set<DAGNode::ptr>::iterator it = dags.begin(); it != dags.end(); ++it)
    {
        const DAGNode* leafNode = (*it)->getLeafNode(featureVector);
        const ClassHistogram* hist = leafNode->getClassHistogram();
        
        int i = leafNode->getClassLabel();
        {
            // If this class has no relevance, don't cast any votes
            if (hist->at(i) <= 0) continue;
            
            // Did we already encounter this class?
            if (votes.find(i) != votes.end())
            {
                votes[i] = votes[i] + 1;
            }
            else
            {
                votes[i] = 1;
            }
        }
    }
    
    // Find the best class
    float bestScore = 0;
    float scoreSum = 0;
    ClassLabel bestLabel = -1;
    
    for (std::map<ClassLabel, float>::iterator it = votes.begin(); it != votes.end(); ++it)
    {
        scoreSum += it->second;
       
        if (it->second > bestScore)
        {
            bestScore = it->second;
            bestLabel = it->first;
        }
    }
    
    return PredictionResult::Factory::create(bestLabel, bestScore/scoreSum);
}

DataPoint::ptr DataPoint::Factory::createFromFileRow(const std::vector<std::string> & _row)
{
    DataPoint::ptr dataPoint = DataPoint::Factory::createZeroInitialized(_row.size());
    
    // Parse the string elements of the vector
    int counter = 0;
    for (std::vector<std::string>::const_iterator it = _row.begin(); it != _row.end(); ++it)
    {
        (*dataPoint)[counter++] = atof(it->c_str());
    }
    
    return dataPoint;
}

DataSet::ptr DataSet::Factory::createFromFile(const std::string & _fileName, bool _verboseMode)
{
    // Create a blank training set and load the file line by line
    DataSet::ptr trainingSet = DataSet::Factory::create();
    
    std::string data(_fileName);

    std::ifstream in(data.c_str());
    if (!in.is_open())
    {
        throw RuntimeException("Could not open data set file.");
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
        std::cout << "Loading data set from " << _fileName << std::endl;
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
        
        // Create the corresponding data point by considering only the last columns
        DataPoint::ptr point = DataPoint::Factory::createFromFileRow(row);
        trainingSet->push_back(point);
    }
    std::cout << "Data set loaded. Number of examples: " << trainingSet->size() << std::endl ;
    
    return trainingSet;
}

void DAGNode::Factory::serialize(DAGNode::ptr node, bool isRoot, std::ofstream & outfile)
{
    /**
     * The serialized model has the following structure
     * 
     * [nodeID], [isRoot], [featureID], [threshold], [left child ID], [right child ID], [class label], "[class histogram]"
     */
    outfile << node->getID() << ',';
    
    if (isRoot)
    {
        outfile << "1,";
    }
    else
    {
        outfile << "0,";
    }
    
    outfile << node->getFeatureID() << ',' <<  node->getThreshold() << ',';
    
    // We only save the histogram and class label at the leaf nodes
    if (node->getLeft() != 0)
    {
        // Don't store a class histogram or a class label
        outfile << node->getLeft()->getID() << ',' << node->getRight()->getID() << ",,,\n";
    }
    else
    {
        // Don't store a child nodes
        // Store the class label and histogram
        outfile << "0,0," << node->getClassLabel() << ",\"";
        
        for (int i = 0; i < node->getClassHistogram()->size(); i++)
        {
            outfile << node->getClassHistogram()->at(i);
            if (i != node->getClassHistogram()->size() - 1)
            {
                outfile << ',';
            }
            else
            {
                outfile << "\"\n";
            }
        }
    }
}

Jungle::ptr Jungle::Factory::createFromFile(const std::string& _filename, bool _verboseMode)
{
    // Try to open the model file
    std::ifstream in(_filename);
    if (!in.is_open())
    {
        throw RuntimeException("Could not open training set file.");
    }

    // Count the number of lines in order to display the progress bar
    std::ifstream countFile(_filename); 
    int lineCount = std::count(std::istreambuf_iterator<char>(countFile), std::istreambuf_iterator<char>(), '\n');
    countFile.close();
    
    ProgressBar::ptr progressBar = ProgressBar::Factory::create(lineCount);
    
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    std::vector< std::string > row;
    std::string line;
    
    // We keep all created nodes under their ID in this map
    std::map<int, DAGNode::ptr> nodes;
    
    Jungle::ptr jungle = Jungle::Factory::create();
    
    while (std::getline(in,line))
    {
        if (_verboseMode)
        {
            progressBar->update();
        }
        
        Tokenizer tok(line);
        row.assign(tok.begin(),tok.end());

        // Do not consider blank line
        if (row.size() < 2) continue;
        
        // Load the training example to the training set
        // Get the node ID
        int nodeID = atoi(row[0].c_str());
        // Get whether or not this is a root node
        bool isRootNode = atoi(row[1].c_str()) == 1;
        
        // Unserialize the node
        DAGNode::ptr node = DAGNode::Factory::unserialize(row);
        nodes[nodeID] = node;
        
        if (isRootNode)
        {
            jungle->getDAGs().insert(node);
        }
    }
    
    // Recover all child node assignments
    for (std::map<int, DAGNode::ptr>::iterator it = nodes.begin(); it != nodes.end(); ++it)
    {
        DAGNode::ptr node = it->second;
        
        // Is this a child node?
        if (node->getTempLeft() == 0)
        {
            // Yep, nothing to do here
            node->setLeft(0);
            node->setRight(0);
        }
        else
        {
            // Nope, recover the pointers
            node->setLeft(nodes[node->getTempLeft()]);
            node->setRight(nodes[node->getTempRight()]);
        }
    }
    
    return jungle;
}

DAGNode::ptr DAGNode::Factory::unserialize(const std::vector<std::string> & row)
{
    // There must be exactly 8 entries in the vector. Otherwise the model is corrupt
    if (row.size() != 8)
    {
        throw RuntimeException("Invalid model row.");
    }
    
    // Row structure
    // 0         1         2            3            4                5                 6              7 
    // [nodeID], [isRoot], [featureID], [threshold], [left child ID], [right child ID], [class label], "[class histogram]"
    // The first two entries don't matter here
    DAGNode::ptr node = DAGNode::Factory::create(0);
    
    node->setFeatureID(atoi(row[2].c_str()));
    node->setThreshold(atof(row[3].c_str()));
    node->setTempLeft(atoi(row[4].c_str()));
    node->setTempRight(atoi(row[5].c_str()));
    
    // Is this a child node?
    if (row[4] == "0")
    {
        // It is, recover the class histogram and class label
        node->setClassLabel(atoi(row[6].c_str()));
        
        // Parse the histogram list (Only comma separated)
        std::vector<int> histogram;
        std::string h = row[7];
        int last = 0;
        for (std::vector<int>::size_type i = 0; i < h.size(); i++)
        {
            if (h[i] == ',')
            {
                // Add the last bin to the histogram
                histogram.push_back(atoi(h.substr(last, i - last).c_str()));
                last = i+1;
            }
        }
        // Add the last entry
        histogram.push_back(atoi(h.substr(last, h.size() - last).c_str()));
        
        // Add the values to the node histogram
        node->getClassHistogram()->resize(histogram.size());
        
        for (std::vector<int>::size_type i = 0; i < histogram.size(); i++)
        {
            node->getClassHistogram()->set(i, histogram[i]);
        }
    }
    
    return node;
}
