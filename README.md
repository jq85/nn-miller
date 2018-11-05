# Neural Net in C++ Tutorial
# Stopped in min 59
Source: https://vimeo.com/19569529
Update: For a newer neural net simulator optimized for image processing, see http://neural2d.net.

Update: For a beginner's introduction to the concepts and abstractions needed to understand how neural nets learn and work, and for tips for preparing training data for your neural net, see the new companion video "The Care and Training of Your Backpropagation Neural Net" at vimeo.com/technotes/neural-net-care-and-training .

Neural nets are fun to play with. Join me as we design and code a classic back-propagation neural net in C++, with adjustable gradient descent learning and adjustable momentum. Then train your net to do amazing and wonderful things. More at the blog: http://www.millermattson.com/dave/

## Output neurons


## Input neurons
Input neurons don't process, they are like latches, they just hold the input values for the whole neural net.
They hold their input value in their output side.

## Hidden neurons
When their input changes, they go through all their input values and the connections weight and change their outputs until the output of the layer changes.
They take the inputs and the weights of the connections to change their outputs.
It is the weights of the connections what form the mathematical transform.
When you train the net it is all about changing the weights until the net solves something.

## Bias neurons
A constant bias input is sometimes useful.
These have only outputs and no inputs (or constant zero).
The weights of their connections might change though.
