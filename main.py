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
EPS = 1  # the starting value of epsilon
EPS_DECAY = 0.999  # controls the rate of exponential decay of epsilon, higher means a slower decay
MIN_EPS = 0.1
GAMMA = 0.99  # Discount Factor
BATCH_SIZE = 128  # is the number of transitions random sampled from the replay buffer
LEARNING_RATE = 1e-3  # is the learning rate of the Adam optimizer, should decrease (1e-5)
steps_done = 0
hidden_sizes = [128, 64]

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


MAX_DISTANCE = 15.0
MIN_DISTANCE = 0.2


def normalize(state: np.ndarray) -> np.ndarray:
    nornmalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
    for i in range(len(state)):
        if i < 17:
            nornmalized_state[i] = (state[i] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        else:
            nornmalized_state[i] = state[i] / math.pi
    return nornmalized_state


# Training loop
if TRAIN:
    env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
    env.action_space.seed(42)  # 42

    state_observation, info = env.reset(seed=42)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    n_observations = len(state_observation)

    # Instantiate policy network.
    q_function = DQLN(n_observations, n_actions).to(device)

    # Instantiate target network.
    target_q_function = DQLN(n_observations, n_actions).to(device)
    target_q_function.load_state_dict(q_function.state_dict())

    # Use Adam to optimize.
    optimizer = optim.Adam(q_function.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    replay_buffer = ReplayMemory(50000)


    # Epsilon-greedy action sampling.
    def select_action_epsilon(state):

        sample = random.random()

        if sample > EPS:
            with torch.no_grad():
                # return index of action with the best Q value in the current state
                return q_function(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    # Do one batch of gradient descent.
    def optimize_model():
        # Make sure we have enough samples in replay buffer.
        if len(replay_buffer) < BATCH_SIZE:
            return

        # Sample uniformly from replay buffer.
        transitions = replay_buffer.sample(BATCH_SIZE)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate into tensors for batch update.
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.Tensor(batch.done)

        # policy_net computes Q(state, action taken)
        state_action_values = q_function(state_batch).gather(1, action_batch)

        # Compute the expected Q values with BELLMAN OPTIMALITY Q VALUE EQUATION:
        # Q(state,action) = reward(state,action) + GAMMA * max(Q(next_state, actions), action)
        expected_state_action_values = reward_batch + GAMMA * (1 - done_batch) * target_q_function(next_state_batch).max(1)[0]

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        writer.add_scalars('loss', {'policy_net': loss}, i_episode)

        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


    def run_validation(env, policy_net, num=10):

        running_rewards = [0.0] * num
        for i in range(num):
            state_observation, info = env.reset()
            while True:
                state_observation = normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)

                action = policy_net(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, _ = env.step(action.item())
                running_rewards[i] += reward

                if terminated or truncated:
                    break
        return running_rewards


    if torch.cuda.is_available():
        num_episodes = 2001
    else:
        num_episodes = 2001

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter("runs")

    print("START Deep Q-Learning Navigation Goal")

    # Sample experience, save in Replay Buffer.
    for i_episode in range(0, num_episodes, 1):
        print("Episode: ", i_episode)
        state_observation, info = env.reset()
        state_observation = normalize(state_observation)
        state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
        steps = 0

        # Run one episode.
        while True:
            action = select_action_epsilon(state_observation)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            observation = normalize(observation)  # Normalize in [0,1]

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            replay_buffer.push(state_observation, action, next_state, reward, done)

            # Move to the next state
            state_observation = next_state
            steps += 1

            if done:
                break

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Epsilon decay
        EPS = max(MIN_EPS, EPS * EPS_DECAY)

        # Update target network.
        if not i_episode % 10:
            policy_net_state_dict = q_function.state_dict()
            target_q_function.load_state_dict(policy_net_state_dict)

        # Every 50 episodes, validate.
        if not i_episode % 50:
            rewards = run_validation(env, q_function)
            writer.add_scalars('Reward', {'policy_net': np.mean(rewards)}, i_episode)
            writer.add_scalars('Epsilon', {
                'policy_net': EPS}, i_episode)

    PATH = './checkpoints/last.pth'
    torch.save(q_function.state_dict(), PATH)
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

    q_function = DQLN(n_observations, n_actions).to(device)
    PATH = './checkpoints/last.pth'
    q_function.load_state_dict(torch.load(PATH))
    not_terminated = 0
    success = 0
    TEST_EPISODES = 500
    for _ in range(TEST_EPISODES):
        steps = 0
        while True:
            state_observation = normalize(state_observation)
            state_observation = torch.tensor(state_observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = q_function(state_observation).max(1)[1].view(1, 1)
            state_observation, reward, terminated, truncated, info = env.step(action.item())
            steps += 1
            if steps >= 200:
                not_terminated += 1
                truncated = True

            if terminated or truncated:
                if not truncated and reward == 500:
                    success += 1
                state_observation, info = env.reset()
                break

    env.close()
    print("Executed " + str(TEST_EPISODES) + " episodes:\n" + str(success) + " successes\n" + str(
        not_terminated) + " episodes not terminated\n" + str(
        TEST_EPISODES - (success + not_terminated)) + " failures\n")
