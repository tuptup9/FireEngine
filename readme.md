# The Mobile(Net) Firefighter Hyper-heuristic: FireEngine

For further information about the solver, the features, the heuristics, and the problem representation, you may check this repository here: https://github.com/jcobayliss/FFPHHS , they are unchanged.

## About the hyper-heuristics

The hyper-heuristic used is a Deep Learning Model using the MobileNetV2 architecture and the Tensorflow/Keras framework. It treats the matrix representation of the graphs from a firefighter problem as an image, and then predicts which heuristic will do best given a graph. To create the dataset, we use the NODES_IN_DANGER feature and the amount of protected nodes as metrics on how good a certain heuristic is given a problem: Whichever heuristic minimizes the amount of nodes that are no longer in danger is given to the model as the heuristic to predict. 

The model is trained using an 95% Train/Validation split using the problems in the instances folder. We have overfitted the model on purpose as it shows poor generalization given the relatively small amount of data used (~300 problems).

- **__init__**. The constructor of your hyper-heuristic. It must receive which heuristics the model will predict. Note that it was trained with LDEG and GDEG, and will assume those heuristics are the input unless you retrain it using main.py first, you can change the heuristics to train in data.py. It does not need to receive features as the network will decide which parts of the matrix are the most important to pay attention to.
- **__str__**. It returns the model summary and the heuristics. 
- **nextHeuristic**- The core of the hyper-heuristic since it indicates the next heuristic to apply given the current problem state. It asks the model to predict given a graph, and then returns an array with the class probabilities. It takes the index of the highest probability and uses it to return the corresponding heuristic.


## Running this code
You will only need to use main.py and ffp.py to use this code. You will find all the dataset operations in data.py, and the logic for generating the metrics in singlerun.py. 

The main requirements are Numpy and Tensorflow.

To train the model, you will first need to generate the dataset. Uncomment line 46 and run the code. It will display the category of each problem as it progresses, then begin training. For later runs, should you adjust any parameters that do not affect the dataset, you can comment this line again.

To evaluate in any given problem, use ffp.py. Set fileName to your desired input and run! Lines 296 and onwards compare the output with the included dummy hyperheuristic.
