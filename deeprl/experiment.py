import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from itertools import count
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from .agents.common import Transition

class LearningParams(object):
    """Wrapper for model parameters to help code readability.

    Args:
        kwargs (dict): Parameters dictionary.
    """    
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

class Experiment(object):
    """Experiment class.

    An Experiment is a full session of exploration of an Agent in an Environment.

    Args:
        env ([gym.envs.atari.atari_env.AtariEnv, gym.wrappers.monitor.Monitor].): Gym environment instance.
        model_class (): 
        params (LearningParams): Collection of training parameters.
        device (torch.device): Device operating computations.
        screen_transform (torchvision.transforms.transforms.Compose): Preprocessing transforms of input images.
        optimizer_class ():
    """
    def __init__(self, 
                 env,
                 model_class,
                 params,
                 device,
                 memory, 
                 screen_transform,
                 optimizer_class=optim.RMSprop):

        self.env = env
        self.model_class = model_class
        self.params = params
        self.device = device
        self.screen_transform = screen_transform

        init_screen, _ = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        self.h = screen_height
        self.w = screen_width

        self.n_actions = self.env.action_space.n
        self.policy_net = self.model_class(h=screen_height, 
                                           w=screen_width, 
                                           outputs = self.n_actions).to(self.device)

        self.target_net = self.model_class(h=screen_height, 
                                           w=screen_width, 
                                           outputs = self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer_class(self.policy_net.parameters())
        self.memory = memory

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * \
            math.exp(-1. * self.steps_done / self.params.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.params.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.params.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_val = loss.data

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss_val
    
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        original_screen = screen.unsqueeze(0).to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        return self.screen_transform(screen).unsqueeze(0).to(self.device), original_screen

    def run(self, num_episodes, output_video_folder=None):

        self.env.reset()
        save_video = output_video_folder is not None

        if save_video:
            Path(output_video_folder).mkdir(exist_ok=True)

        returns = []
        episode_lenghts = []
        mean_losses = []
        
        for episode_idx in tqdm.tqdm(range(num_episodes)):
            # Initialize the environment and state
            self.env.reset()
            last_screen, _ = self.get_screen()
            current_screen, _ = self.get_screen()
            state = current_screen - last_screen

            for _ in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen, original_screen = self.get_screen()

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss_val = self.optimize_model()

                if done:

                    break
            # Update the target network, copying all weights and biases in DQN
            if episode_idx % self.params.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.env.close()
