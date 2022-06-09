# The Firefighter Problem Hyper-heuristic Solver (FFPHHS)

For further information about the solver, the features, the heuristics, and the problem representation, you may check this repository here: https://github.com/jcobayliss/FFPHHS , they are unchanged.

## About the hyper-heuristics

The hyper-heuristic used is a Deep Learning Model using the MobileNetV2 architecture and the Tensorflow/Keras framework. It treats the matrix representation of the graphs from a firefighter problem as an image, and then predicts which heuristic will do best given a graph. To create the dataset, we use the NODES_IN_DANGER feature as a metric on how good a certain heuristic is given a problem: Whichever heuristic minimizes the amount of nodes in danger in a step is given to the model as the heuristic to predict. 

The model is trained using an 80/10/10 Train/Validation/Test split using the problems in the instances folder. 

- **__init__**. The constructor of your hyper-heuristic. It must receive which heuristics the model will predict. Note that it was trained with LDEG and GDEG, and will assume those heuristics are the input unless you retrain it using main.py first, you can change the heuristics to train in data.py. It does not need to receive features as the network will decide which parts of the matrix are the most important to pay attention to.
- **__str__**. It returns the model summary and the heuristics. 
- **nextHeuristic**- The core of the hyper-heuristic since it indicates the next heuristic to apply given the current problem state. It asks the model to predict given a graph, and then returns an array with the class probabilities. It takes the index of the highest probability and uses it to return the corresponding heuristic.

