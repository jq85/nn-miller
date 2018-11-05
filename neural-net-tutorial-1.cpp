// neural-net-tutorial.cpp
// ANSI ISO Standard compliance
#include <vector>
#include <iostream>

class Neuron {};

typedef std::vector<Neuron> Layer;

/*
*
*/
class Net
{
public:
  Net(const std::vector<unsigned> &topology);// promising not to change the topology
  void feedForward(const std::vector<double> &inputVals) {};// promising not to change the input of values
  void backProp(const std::vector<double> &targetVals) {};// promising not to change the input of values
  void getResults(std::vector<double> &resultVals) const {}; // search on internet for "const correctness". it does not modify the input at all here. We are going to fill in values into the vector of values

private:
  std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum] this container groups all the neurons for all the layers
};

Net::Net(const std::vector<unsigned> &topology)
{
  unsigned numLayers = topology.size();
  for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
  {
    // in each iteration we want to create a new layer object and add it to the m_n layer container
    m_layers.push_back(Layer());

    // we have made a new layer, now fill it with neurons, and
    // add a bias neuron to the layer:
    for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum ) // we want one extra bias neuron, that's why <=, and not <
    {
      m_layers.back().push_back(Neuron());// .back() gives the last (most recent) element of the container
      std::cout << "Made a Neuron!" << std::endl;//shall output 3in+1bias, 2hidden+1bias, 1out+1bias = 9 neurons in total
    }
  }
};


/*
*
*/
int main()
{
  // e.g., { 3, 2, 1 }
  std::vector<unsigned> topology;
  topology.push_back(3);// The first layer will have 3 neurons
  topology.push_back(2);// The second layer will have 2 neurons
  topology.push_back(1);// The third layer will have 1 neuron
  Net myNet(topology);


  // TRAINING
  // below the whole basic public API is used
  // to train it
  std::vector<double> inputVals;
  myNet.feedForward(inputVals);
  // during training, after feeding forward, we have to tell it what the outputs are supposed to have been,
  // so it can go through and do its Backpropagation learning.
  // structure of target output values
  std::vector<double> targetVals;
  myNet.backProp(targetVals);

  std::vector<double> resultVals;
  myNet.getResults(resultVals);

}
