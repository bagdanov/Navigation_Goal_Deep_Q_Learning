import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gym_navigation.memory.replay_memory import ReplayMemory, Transition
import numpy as np
import math
import random

TRAIN = False
EPS_START = 0.9  # the starting value of epsilon
EPS_END = 0.3  # the final value of epsilon
EPS_DECAY = 60000  # controls the rate of exponential decay of epsilon, higher means a slower decay
GAMMA = 0.99  # Discount Factor
BETA = 0.005  # is the update rate of the target network
BATCH_SIZE = 128  # is the number of transitions random sampled from the replay buffer
LEARNING_RATE = 1e-4  # is the learning rate of the Adam optimizer, should decrease (1e-5)
steps_done = 0
hidden_sizes = [128, 64]

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print("DEVICE:", device)

class DQLN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQLN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


SCAN_RANGE_MAX = 15.0
SCAN_RANGE_MIN = 0.2
MAX_DISTANCE = 15.0
MIN_DISTANCE = 0.2


def normalize(state: np.ndarray) -> np.ndarray:
    nornmalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
    for i in range(len(state)):
        if i < 16:
            nornmalized_state[i] = (state[i] - SCAN_RANGE_MIN) / (SCAN_RANGE_MAX - SCAN_RANGE_MIN)
        elif i == 16:
            nornmalized_state[i] = (state[i] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        else:
            nornmalized_state[i] = state[i] / math.pi
    return nornmalized_state


if TRAIN:

    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
    env.action_space.seed(42)  # 42

    state_observation, info = env.reset(seed=42)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    n_observations = len(state_observation)

    policy_net = DQLN(n_observations, n_actions).to(device)

    # COMMENT FOR INITIAL TRAINING
    # PATH = '../neural_network/Base.pth'
    # policy_net.load_state_dict(torch.load(PATH))

    target_net = DQLN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    replay_buffer = ReplayMemory(6000)


    def select_action_epsilon(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # return index of action with the best Q value in the current state
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    def optimize_model():
        if len(replay_buffer) < BATCH_SIZE:
            return
        transitions = replay_buffer.sample(BATCH_SIZE)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # map() function returns a map object of the results after applying the given function to each item of a given iterable
        non_final_states_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                             dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # policy_net computes Q(state, action taken)
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # target_net computes max over actions of Q(next_state, action) for all next states
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_states_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values with BELLMAN OPTIMALITY Q VALUE EQUATION:
        # Q(state,action) = reward(state,action) + GAMMA * max(Q(next_state, actions), action)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


    if torch.cuda.is_available():
        num_episodes = 2001
    else:
        num_episodes = 10

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter("runs")

    print("START Deep Q-Learning Navigation Goal")

    for i_episode in range(0, num_episodes, 1):
        print("Episode: ", i_episode)
        state_observation, info = env.reset()
        state_observation = normalize(state_observation)
        state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
        steps = 0
        while True:
            action = select_action_epsilon(state_observation)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            observation = normalize(observation)  # Normalize in [0,1]
            if steps >= 100:
                if not terminated:
                    reward = -100
                    truncated = True
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated or truncated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            replay_buffer.push(state_observation, action, next_state, reward)

            # Move to the next state
            state_observation = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            steps += 1

            # Soft update of the target network's weights
            # Q′ ← β * Q + (1 − β) * Q′
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key] * BETA + target_net_state_dict[key] * (1 - BETA)
            # target_net.load_state_dict(target_net_state_dict)

            if done:
                policy_net_state_dict = policy_net.state_dict()
                target_net.load_state_dict(policy_net_state_dict)
                break

        if i_episode % 50 == 0:
            state_observation, info = env.reset()
            steps = 0
            while True:
                state_observation = normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, _ = env.step(action.item())
                steps += 1
                if steps >= 100:
                    truncated = True

                if terminated or truncated:
                    # tensorboard --logdir=runs
                    writer.add_scalars('Reward', {'policy_net': reward}, i_episode)
                    break

    PATH = '../neural_network/last.pth'
    torch.save(policy_net.state_dict(), PATH)
    writer.close()
    env.close()
    print('COMPLETE')

else:

    # For accuracy check
    # env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
    # For visible check
    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode='human', track_id=1)

    env.action_space.seed(42)
    state_observation, info = env.reset(seed=42)

    n_actions = env.action_space.n
    n_observations = len(state_observation)

    policy_net = DQLN(n_observations, n_actions).to(device)
    PATH = '../neural_network/Best.pth'
    policy_net.load_state_dict(torch.load(PATH))
    not_terminated = 0
    success = 0
    TEST_EPISODES = 100
    for _ in range(TEST_EPISODES):
        steps = 0
        while True:
            state_observation = normalize(state_observation)
            state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_observation).max(1)[1].view(1, 1)
            state_observation, reward, terminated, truncated, info = env.step(action.item())
            steps += 1
            if steps >= 200:
                not_terminated += 1
                truncated = True

            if terminated or truncated:
                if not truncated and info["result"] == "Goal_Reached":
                    success += 1
                state_observation, info = env.reset()
                break

    env.close()
    print("Executed " + str(TEST_EPISODES) + " episodes:\n" + str(success) + " successes\n" + str(
        not_terminated) + " episodes not terminated\n" + str(
        TEST_EPISODES - (success + not_terminated)) + " failures\n")
