import torch
import gym
import numpy as np
import torchvision.transforms as trans

import matplotlib.pyplot as plt
from collections import namedtuple

SARSD = namedtuple('SARSD', ['state', 'action', 'reward', 'next_state', 'done'])


class GymRoom:
    
    def __init__(self, env_type):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.env = gym.make(env_type).unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):        
        state, action, reward, next_state, done = self.env.step(action.item())
        return SARSD(state=state, action=action, reward=reward, next_state=next_state, done=done)
        
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_dim(self):
        # return width, height
        screen = self.get_processed_screen()
        return screen.shape[3], screen.shape[2]
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
        
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = trans.Compose([
            trans.ToPILImage()
            ,trans.Resize((40,90))
            ,trans.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW)
    
    
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gym_room = GymRoom('CartPole-v0', device)
    gym_room.reset()
    screen = gym_room.get_processed_screen()

    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1, 2, 0))
    plt.title('processed screen example')
    plt.show()
