// neural-net-tutorial.cpp
// ANSI ISO Standard compliance
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

struct Connection
{
  double weight;
  double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

/************************ class Neuron ****************************************
*
*/
class Neuron
{
public:
  //needs to construct the vector of connections, so it needs to know the number of connections to make
  // the minimum ammount of information is the number of neurons in the next layer
  // the number of outputs is enough
  Neuron(unsigned numOutputs, unsigned myIndex);
  void setOutputVal(double val) { m_outputVal = val; };
  double getOutputVal(void) const { return m_outputVal; };
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);

private:
  static double eta;// [0.0..1.0] overall net training rate; Only needed by Neuron class, thus static
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return rand() / double(RAND_MAX); }
  double sumDOW(const Layer &nextLayer) const;
  double m_outputVal;
  std::vector<Connection> m_outputWeights;
  unsigned m_myIndex;
  double m_gradient;
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]



void Neuron::updateInputWeights(Layer &prevLayer)
{
  // The weights to be updated are in the Connection container
  // in the neurons in the previous layer
  for(unsigned n = 0; n < prevLayer.size(); ++n)
  {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight =
        // individual input, magnified by the gradient and train rate:
        eta
        * neuron_getOutputVal()
        * m_gradient
        // Also add momentum = a fraction of the previous delta weight
        + alpha
        * oldDeltaWeight;
    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}


double Neuron::sumDOW(const Layer &nextLayer) const
{
  double sum = 0.0;

  // Sum our contributions of the errors at the nodes we feed
  for(unsigned n = 0; n < nextLayer.size() - 1; ++n)
  {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }
  return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);// one of many ways to calc gradient; this reduces the error
}

double Neuron::transferFunction(double x)
{
  // better a curve, a sine, etc, but could have higher cpu cost
  // tanh - output range [ -1.0..1.0]
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
  // tanh derivative
  return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.
  for(unsigned n = 0; n < prevLayer.size(); ++n)
  {
    sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
  }

  // shape the output, transfer, or activation function
  m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
  for(unsigned connections = 0; connections < numOutputs; ++connections)
  {
    m_outputWeights.push_back(Connection());
    // set the weight to something random; it could be done on the constructor
    m_outputWeights.back().weight = randomWeight();
  }

  m_myIndex = myIndex;
}


/************************ class Net *******************************************
*
*/
class Net
{
public:
  Net(const std::vector<unsigned> &topology);// promising not to change the topology
  void feedForward(const std::vector<double> &inputVals);// promising not to change the input of values
  void backProp(const std::vector<double> &targetVals);// promising not to change the input of values
  void getResults(std::vector<double> &resultVals) const; // search on internet for "const correctness". it does not modify the input at all here. We are going to fill in values into the vector of values

private:
  std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum] this container groups all the neurons for all the layers
  double m_error;
  double m_recentAverageError;
  double m_recentAverageSmoothingFactor;
};

void Net::getResults(std::vector<double> &resultVals) const
{
  resultVals.clear();
  for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
  {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

void Net::backProp(const std::vector<double> &targetVals)
{
  // Calculate overall net error (RMS --Root Mean Square-- of output neuron errors)
  Layer &outputLayer = m_layers.back();
  m_error = 0.0;
  for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
  {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; // get average error squared
  m_error = sqrt(m_error); // RMS

  // Implement a recent average measurement:
  m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
  {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate gradients on hidden layers
  for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
  {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];

    for(unsigned n = 0; n < hiddenLayer.size(); ++n)
    {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }

  // For all layers from outputs to first hidden layer,
  // update connection weights
  for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
  {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];
    for(unsigned n = 0; n < layer.size() - 1; ++n)
    {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
  // assert is useful for rapid prototyping
  assert(inputVals.size() == m_layers[0].size() - 1);// check that input neurons equals number of elements in inputVals (-1 bias neuron)

  // Assign (latch) the input values into the input neurons
  for(unsigned i=0; i < inputVals.size(); ++i)
  {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }

  // Forward propagate, telling each neuron to feed forward
  for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
  {
    Layer &prevLayer = m_layers[layerNum - 1];//&prevLayer to make it faster as a pointer
    for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
    {
      m_layers[layerNum][n].feedForward(prevLayer);// feed forward
    }
  }
}

Net::Net(const std::vector<unsigned> &topology)
{
  unsigned numLayers = topology.size();
  for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
  {
    // in each iteration we want to create a new layer object and add it to the m_n layer container
    m_layers.push_back(Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

    // we have made a new layer, now fill it with neurons, and
    // add a bias neuron to the layer:
    for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum ) // we want one extra bias neuron, that's why <=, and not <
    {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));// .back() gives the last (most recent) element of the container
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
