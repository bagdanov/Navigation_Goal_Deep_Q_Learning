# Navigation_Goal_Deep_Q_Learning
Fundamentals of Machine Learning Laboratory Exam

In this project an Agent has been trained to overcome a Navigation Goal task. This was achieved by using a Multilayer Perceptron for learing to reproduce Bellman optimality equation for the action-value function.

In the gym_navigation/envs directory you will find three python modules that define the behavior of the agent and the Environment in which it operates. These are based on the Gymnasium library and also use the modules defined in the gym_navigation/enums and gym_navigation/geometry directories for their purpose. Gym_navigation/enums contains the enumerations that define the possible actions of the Agent, the colors used for video rendering (via the Pygame library) and the size and shape of the Environment box. In gym_navigation/geometry there are instead three modules that define classes and methods useful for managing the various objects present in the Environment.

In script/main.py you can find the core of the project. Changing the variable "TRAIN" to True you will be able to train your Agent  


LSTM for Planning

Increase Action range of Agent

Increase the number of episodes for Training
