# Navigation_Goal_Deep_Q_Learning


This is a simple 2D navigation environment implemented as a part of
the *Fundamentals of Machine Learning Laboratory*. 

In the `gym_navigation/envs` directory you will find three Python
modules that define the behavior of the agent and the environment in
which it operates. The `navigation.py` file contains the abstract
class of our world, whereas the other two files contain the concrete
classes that implement the parent abstract methods specializing the
agent in ray casting tracking and in the navigation goal task. These
modules are based on the [Gymnasium library](https://gymnasium.farama.org/) and also use the modules
defined in the `gym_navigation/enums` and `gym_navigation/geometry`
directories for their purpose. The directory `gym_navigation/enums` contains the
enumerations that define the possible actions of the Agent, the colors
used for rendering (via the PyGame library) and the size and
shape of the environment box. In `gym_navigation/geometry` there are
instead three modules that define classes and methods useful for
managing the various objects present in the environment.

This repository also contains an implementation of Deep Q-Learning
used to train an agent has been trained to overcome the Navigation
Goal task. This agent is a Multilayer Perceptron (MLP), and in
`main.py` you will its definition (in PyTorch) and many configuration variables
for the parameters for the Q-Learning training process. Changing the
value of the variable `TRAIN` to `True` you can to train your
agent from scratch or load a pretrained net and continue the training for
improving the results. Setting `TRAIN` to `False` will test
a pretrained agent.

During training the library Tensorboard is used for saving the
validation test results. Tensorboard writes to the `runs/` directory
by default. At the end of every training run the agent network is
saved in `checkpoints/last.pth`. In the `gym_navigation/memory`
directory is a class that defines the buffer for storing the agent's
observations during the training process.

In the `checkpoints/` directory you will find two ".pth" files, one is
the best net which I was able to train, while the other is a starting
point for a training session. Set as you want all the various
parameters in `main.py`, `gym_navigation/envs/navigation_goal.py`
and in `gym_navigation/envs/navigation_track.py` and start training your
agents. Enjoy!

For more information take a look at the `presentation/Presentazione_Laboratorio_FOML.pdf`.

Some improvements to try:
- Use a Long Short Term Memory (LSTM) instead of the MLP network for a hopefully better agent future planning
- Increase the agent's possible actions and mobility
- Increase the number of training episodes
- Improve Q-Learning stability
