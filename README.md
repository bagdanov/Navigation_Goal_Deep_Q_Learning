# Navigation_Goal_Deep_Q_Learning
Fundamentals of Machine Learning Laboratory Exam

In this project an Agent has been trained to overcome a Navigation Goal task. This was achieved by using a Multilayer Perceptron (MLP) for learing to reproduce Bellman optimality equation for the action-value function.

In the gym_navigation/envs directory you will find three python modules that define the behavior of the agent and the Environment in which it operates. The navigation.py file contains the abstract class of our world, whereas the other two files contain the concrete classes that implement the parent's abstract methods specializing the Agent in ray casting tracking and in navigation goal task. These modules are based on the Gymnasium library and also use the modules defined in the gym_navigation/enums and gym_navigation/geometry directories for their purpose. Gym_navigation/enums contains the enumerations that define the possible actions of the Agent, the colors used for video rendering (via the Pygame library) and the size and shape of the Environment box. In gym_navigation/geometry there are instead three modules that define classes and methods useful for managing the various objects present in the Environment.

In script/main.py file you can find the MLP net definition (via PyTorch library) and a lot of variables representing the parameters for our training process. Changing the value of the variable "TRAIN" to True you will be able to train your Agent from zero or load a pretrained net and continue the training for improving the results. Whereas setting "TRAIN" to False you can test your Agent for checking its growth. During the training the library Tensorboard is used for saving the validation test results for analyse the andament of the Agent. You will find them in script/runs directory by default. At the end of every training process the net is saved in neural_network/last.pth file. In "memory" directory is contained the class that defines the buffer for memorize the Agent observations during the training process. 


LSTM for Planning

Increase Action range of Agent

Increase the number of episodes for Training
