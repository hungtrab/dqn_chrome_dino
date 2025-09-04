from abc import ABC
from collections import deque
import enum
import os
from typing import Any
from PIL import Image
import gymnasium as gym
import numpy as np
import pygame.freetype
import pygame
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import argparse
import torch
import cv2
from model import DQN

# Environment's constants
WINDOW_SIZE = (1024, 512)  # (w, h)
# It took (JUMP_DURATION / 2) to jump to the peak and another (JUMP_DURATION / 2) to fall to the ground
JUMP_DURATION = 12
JUMP_VEL = 100
OBSTACLE_MIN_CNT = 400
MAX_SPEED = 100
MAX_CACTUS_SPAWN_PROB = 0.7
BASE_CACTUS_SPAWN_PROB = 0.3
BIRD_SPAWN_PROB = 0.3
RENDER_FPS = 15
COLLISION_THRESHOLD = 20
DIFFICULTY_INCREASE_FREQ = 20


class Action(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class DinoState(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class GameMode(str, enum.Enum):
    NORMAL = "normal"
    # In the train mode, when the agent collide with obstacles,
    # it gets negative rewards instead of losing the game.
    TRAIN = "train"


class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"


class Assets:
    def __init__(self):
        # running track
        self.track = pygame.image.load(os.path.join("assets", "Track.png"))

        # dino
        self.dino_runs = [
            pygame.image.load(os.path.join("assets", "DinoRun1.png")),
            pygame.image.load(os.path.join("assets", "DinoRun2.png")),
        ]
        self.dino_ducks = [
            pygame.image.load(os.path.join("assets", "DinoDuck1.png")),
            pygame.image.load(os.path.join("assets", "DinoDuck2.png")),
        ]
        self.dino_jump = pygame.image.load(os.path.join("assets", "DinoJump.png"))

        # cactus
        self.cactuses = [
            pygame.image.load(os.path.join("assets", "LargeCactus1.png")),
            pygame.image.load(os.path.join("assets", "LargeCactus2.png")),
            pygame.image.load(os.path.join("assets", "LargeCactus3.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus1.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus2.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus3.png")),
        ]

        # bird
        self.birds = [
            pygame.image.load(os.path.join("assets", "Bird1.png")),
            pygame.image.load(os.path.join("assets", "Bird2.png")),
        ]


class EnvObject(ABC):
    rect: pygame.Rect

    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Obstacle(EnvObject, ABC):
    # A flag indicates if the agent already passes or collides the obstacle.
    # This is used to avoid "duplicating" rewards for passing/colliding an obstacle.
    needs_collision_check = True

    def collide(self, o: pygame.Rect) -> bool:
        return self.rect.colliderect(
            o.left + COLLISION_THRESHOLD,
            o.top + COLLISION_THRESHOLD,
            o.width - 2 * COLLISION_THRESHOLD,
            o.height - 2 * COLLISION_THRESHOLD,
        )

    def is_inside(self) -> bool:
        return False


class Bird(Obstacle):
    def __init__(self, assets: Assets):
        self._assets = assets.birds
        self.rect = self._assets[0].get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = 360

    def step(self, speed: int):
        self.rect.x -= speed
        # Alternate the assets to create a moving animation
        self._assets[0], self._assets[1] = (
            self._assets[1],
            self._assets[0],
        )

    def is_inside(self) -> bool:
        return self.rect.x + self._assets[0].get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._assets[0],
            self.rect,
        )


class Cactus(Obstacle):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self.rect = self._asset.get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = WINDOW_SIZE[1] - self._asset.get_height() - 7

    def step(self, speed: int):
        self.rect.x -= speed

    def is_inside(self) -> bool:
        return self.rect.x + self._asset.get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            self.rect,
        )


class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = assets.dino_runs
        self._duck_assets = assets.dino_ducks
        self._jump_asset = assets.dino_jump

        self._jump_timer = 0
        self.state = DinoState.STAND

    def step(self, action: Action):
        # Alternate the assets to create a moving animation
        self._run_assets[0], self._run_assets[1] = (
            self._run_assets[1],
            self._run_assets[0],
        )
        self._duck_assets[0], self._duck_assets[1] = (
            self._duck_assets[1],
            self._duck_assets[0],
        )

        # Check if the jump animation is finished
        if self.state == DinoState.JUMP:
            self._jump_timer -= 1
            if self._jump_timer < 0:
                self.state = DinoState.STAND

        # If dino is not jumping, transition to a new state based on the action
        if self.state != DinoState.JUMP:
            match action:
                case Action.STAND:
                    self.state = DinoState.STAND
                case Action.JUMP:
                    self.state = DinoState.JUMP
                    self._jump_timer = JUMP_DURATION
                case Action.DUCK:
                    self.state = DinoState.DUCK

    def get_data(self) -> tuple[pygame.Surface, pygame.Rect]:
        match self.state:
            case DinoState.STAND:
                asset = self._run_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()
            case DinoState.JUMP:
                asset = self._jump_asset
                y = WINDOW_SIZE[1] - self._get_jump_offset() - asset.get_height()
            case DinoState.DUCK:
                asset = self._duck_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()

        rect = pygame.Rect(50, y, asset.get_width(), asset.get_height())

        return asset, rect

    def _get_jump_offset(self) -> int:
        a = -JUMP_VEL / (JUMP_DURATION / 2)
        t = JUMP_DURATION - self._jump_timer
        # Compute the jump distance from acceleration, initial speed, and time
        d = int(JUMP_VEL * t + 0.5 * a * (t**2))
        return d

    def render(self, canvas: pygame.Surface):
        asset, rect = self.get_data()
        canvas.blit(asset, rect)


class Track(EnvObject):
    def __init__(self, assets: Assets):
        self._asset = assets.track

        self._track_offset_x = 0
        self._track_w = self._asset.get_width()
        self._track_h = self._asset.get_height()

    def step(self, speed: int):
        # Negative offset means moving the running track image to the left
        self._track_offset_x -= speed

    def render(self, canvas: pygame.Surface):
        # Render the running track image moved to the left by `track_offset_x`
        canvas.blit(
            self._asset,
            (self._track_offset_x, WINDOW_SIZE[1] - self._track_h),
        )

        # If the moved image doesn't cover the screen, render the left space
        # with a second image to create a "loop" effect.
        if self._track_offset_x + self._track_w < WINDOW_SIZE[0]:
            # Find the starting position to render the second image
            # -10 here because the running track image starts with a small gap
            start_x = self._track_offset_x + self._track_w - 10
            canvas.blit(
                self._asset,
                (start_x, WINDOW_SIZE[1] - self._track_h),
            )

            # If the starting position is negative, which means the moved image
            # doesn't intersect with the screen, start rendering a new image with
            # a new offset equal to the starting position
            if start_x <= 0:
                self._track_offset_x = start_x


class Env(gym.Env):
    metadata = {
        "render_fps": RENDER_FPS,
        "render_modes": [RenderMode.HUMAN, RenderMode.RGB],
    }

    def __init__(
        self,
        render_mode: RenderMode | None,
        game_mode: GameMode = GameMode.NORMAL,
        train_frame_limit=500,  # the upper limit for number of frames during the train mode
    ) -> None:
        # Initialize `gym.Env` required fields
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(list(Action)))
        # The observation space is the dimension of the current frame (rgb image)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8
        )

        self._game_mode = game_mode
        self._train_frame_limit = train_frame_limit

        # Initialize `pygame` data
        self._window = None
        self._clock = None

        pygame.freetype.init()
        self._game_font = pygame.freetype.SysFont(
            pygame.freetype.get_default_font(), 24
        )
        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()

        self._init_game_data()

        super().__init__()

    def _init_game_data(self):
        self._assets = Assets()

        """Initialize game's data, which should be re-initialized when the environment is reset"""
        self._frame = 0
        self._speed = 20
        self._spawn_prob = BASE_CACTUS_SPAWN_PROB
        # The counter (in pixels) for spawning a new obstacle
        self._obstacle_cnt = OBSTACLE_MIN_CNT

        # Initialize environment's objects' states
        self._track = Track(self._assets)
        self._agent = Dino(self._assets)
        self._obstacles: list[Obstacle] = []

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self._init_game_data()

        obs = self._render_frame()

        return obs, {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        reward = 0.0

        self._frame += 1
        self._obstacle_cnt += self._speed
        # Increase the difficulty of the game every fixed number of frames
        if self._frame % DIFFICULTY_INCREASE_FREQ == 0:
            self._speed = min(MAX_SPEED, self._speed + 1)
            self._spawn_prob = min(MAX_CACTUS_SPAWN_PROB, self._spawn_prob * 1.01)

        self._track.step(self._speed)
        self._agent.step(action)
        for o in self._obstacles:
            o.step(self._speed)

        # Filter out outside obstacles after each step
        self._obstacles = [o for o in self._obstacles if o.is_inside()]

        # Check if the agent collides with an obstacle
        _, agent_rect = self._agent.get_data()
        for o in self._obstacles:
            if not o.needs_collision_check:
                continue
            if o.collide(agent_rect):
                o.needs_collision_check = False
                reward -= 1.0
                if self._game_mode == GameMode.NORMAL:
                    terminated = True
            else:
                # Agent passes an obstacle without colliding with the object, give a reward
                if agent_rect.left > o.rect.right:
                    o.needs_collision_check = False
                    reward += 1.0

        if self._game_mode == GameMode.TRAIN and self._frame >= self._train_frame_limit:
            terminated = True

        # Should we spawn a new obstacle?
        self._spawn_obstacle_maybe()

        obs = self._render_frame()

        return obs, reward, terminated, False, {}

    def _spawn_obstacle_maybe(self):
        if self._obstacle_cnt > max(OBSTACLE_MIN_CNT, JUMP_DURATION * self._speed):
            if self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[
                0
            ]:
                id = self.np_random.choice(len(self._assets.cactuses), 1)[0]
                self._obstacles.append(Cactus(self._assets, id))

            elif self.np_random.choice(2, 1, p=[0.9, 0.1])[0]:
                self._obstacles.append(Bird(self._assets))

            self._obstacle_cnt = 0

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        canvas = pygame.Surface(WINDOW_SIZE)
        canvas.fill((255, 255, 255))

        self._track.render(canvas)
        self._agent.render(canvas)
        for o in self._obstacles:
            o.render(canvas)

        # Display the current scores (number of frames)
        text_surface, _ = self._game_font.render(f"score: {self._frame}", (0, 0, 0))
        canvas.blit(text_surface, (10, 10))

        if self._window is not None and self._clock is not None:
            self._window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata["render_fps"])

        # Return the canvas as a rgb array
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def get_dino_state(self):
        """Public method to get the dino's current state"""
        return self._agent.state

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


class Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, k=4, image_size=(128, 64)):
        super().__init__(env)

        self.env = env
        self.k = k
        self.image_size = image_size

        obs_space = env.observation_space.shape
        assert obs_space is not None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.k, self.image_size[1], self.image_size[0]),
            dtype=np.uint8,
        )

        self.frames: list[np.ndarray] = []
        self.stack = deque([], maxlen=self.k)

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        # Convert the observation image from the environment to
        # gray scale and resize it to a corresponding size
        return np.array(
            Image.fromarray(obs).convert("L").resize(self.image_size), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        # Stack the last "k" frames into a single "np.ndarray"
        assert len(self.stack) == self.k
        return np.stack(self.stack)

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self.frames = []
        self.stack = deque([], maxlen=self.k)

        obs, _ = self.env.reset(*args, **kwargs)
        self.frames.append(obs)
        obs = self._transform(obs)

        for _ in range(self.k):
            self.stack.append(obs)

        return self._get_obs(), {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False

        for _ in range(self.k):
            obs, reward, term, *_ = self.env.step(action)
            self.frames.append(obs)
            obs = self._transform(obs)

            self.stack.append(obs)

            total_reward += float(reward)
            if term:
                terminated = True
                break

        return self._get_obs(), total_reward, terminated, False, {}


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

register(
    id="Dino-v0",
    entry_point="dino:Env",
    max_episode_steps=2000,  # Increased limit for longer AI play sessions
)

def preprocess_observation(obs, device):
    """
    Preprocess the observation from Chrome Dino environment
    Expected input: RGB image of shape (height, width, 3)
    Output: Grayscale tensor of shape (1, height, width) for Conv2D
    """
    # If obs is already a tensor, convert to numpy
    if torch.is_tensor(obs):
        obs = obs.cpu().numpy()
    
    # Handle different input shapes
    if len(obs.shape) == 5:  # [batch, 1, H, W, 3] - remove extra dimensions
        obs = obs[0, 0]  # Take first batch and first element
    elif len(obs.shape) == 4:  # [1, H, W, 3] - remove batch dimension
        obs = obs[0]
    elif len(obs.shape) == 3:  # [H, W, 3] - correct shape
        pass
    elif len(obs.shape) == 2:  # Already grayscale [H, W]
        pass
    else:
        raise ValueError(f"Unexpected observation shape: {obs.shape}")
    
    # Convert to grayscale if it's RGB
    if len(obs.shape) == 3 and obs.shape[-1] == 3:
        obs = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Resize to smaller size for faster processing (84x84 is common for Atari games)
    obs = cv2.resize(obs, (84, 84))
    
    # Normalize to [0, 1]
    obs = obs.astype(np.float32) / 255.0
    
    # Add channel dimension: (H, W) -> (1, H, W)
    obs = np.expand_dims(obs, axis=0)
    
    # Create tensor and ensure it's always [1, 84, 84]
    tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # Final safety check - ensure shape is exactly [1, 84, 84]
    if tensor.shape != (1, 84, 84):
        raise ValueError(f"Preprocessing failed: expected (1, 84, 84), got {tensor.shape}")
    
    return tensor

def human_play():
    env = gym.make("Dino-v0", render_mode="human")
    obs, _ = env.reset()

    total_reward = 0.0
    n_frames = 0
    while True:
        n_frames += 1
        userInput = pygame.key.get_pressed()
        action = Action.STAND
        if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
            action = Action.JUMP
        elif userInput[pygame.K_DOWN]:
            action = Action.DUCK

        obs, reward, terminated, _, _ = env.step(action)

        total_reward += float(reward)
        if terminated:
            break

    print(f"Total reward: {total_reward}, number of frames: {n_frames}")

    env.close()

    # Show image of the last frame
    plt.imshow(obs)
    plt.show()


def play_with_model_wrapped(
    env: Wrapper,  # Use wrapped env
    policy_net: DQN,
    device: torch.device,
    seed: int | None = None,
) -> float:
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    # Convert wrapped environment output to tensor
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # Normalize to [0, 1] if not already normalized
        if state.max() > 1.0:
            state = state / 255.0

    total_reward = 0.0
    step_count = 0
    while True:
        step_count += 1
        with torch.no_grad():
            # Add batch dimension for network input
            if len(state.shape) == 3:  # [C, H, W] -> [1, C, H, W]
                state_batch = state.unsqueeze(0)
            else:
                state_batch = state
                
            action_values = policy_net(state_batch)
            action = action_values.max(dim=1)[1][0].item()
            
            # Debug output to understand the behavior
            action_names = ['STAND', 'JUMP', 'DUCK']
            if step_count <= 20 or step_count % 20 == 0:  # Show first 20 steps and every 20th step
                print(f"Step {step_count}: Q-values {action_values.detach().cpu().numpy()}, Action: {action} ({action_names[action]})")

        # Debug: Print what action is actually being sent to the environment
        if step_count <= 20 or step_count % 20 == 0:
            print(f"Step {step_count}: Sending action {action} (type: {type(action)}) to environment")
            
        # Ensure action is converted to Action enum
        action_enum = Action(action)
        next_state, reward, terminated, _, _ = env.step(action_enum)
        
        # Convert next state to tensor
        if isinstance(next_state, np.ndarray):
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
            # Normalize to [0, 1] if not already normalized
            if state.max() > 1.0:
                state = state / 255.0
        else:
            state = next_state

        total_reward += float(reward)
        
        # Debug reward changes
        if reward != 0.0:
            print(f"Step {step_count}: Reward changed to {reward}, Total: {total_reward}")
        
        if terminated:
            print(f"Game terminated at step {step_count} with final reward: {reward}")
            break

    print(f"Game ended after {step_count} steps")
    return total_reward


def play_with_model(
    env: gym.Env,  # Use basic env, not wrapped
    policy_net: DQN,
    device: torch.device,
    seed: int | None = None,
) -> float:
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    state = preprocess_observation(state, device)

    total_reward = 0.0
    step_count = 0
    while True:
        step_count += 1
        with torch.no_grad():
            action_values = policy_net(state.unsqueeze(0))
            action = action_values.max(dim=1)[1][0].item()
            
            # Debug output to understand the behavior
            action_names = ['STAND', 'JUMP', 'DUCK']
            if step_count <= 20 or step_count % 20 == 0:  # Show first 20 steps and every 20th step
                print(f"Step {step_count}: Q-values {action_values.detach().cpu().numpy()}, Action: {action} ({action_names[action]})")

        # Debug: Print what action is actually being sent to the environment
        if step_count <= 20 or step_count % 20 == 0:
            print(f"Step {step_count}: Sending action {action} (type: {type(action)}) to environment")
            
        # Ensure action is converted to Action enum
        action_enum = Action(action)
        state, reward, terminated, _, _ = env.step(action_enum)
        
        # Debug: Check what the dino's actual state is after the action
        if step_count <= 20 or step_count % 20 == 0:
            # Access the dino state through the public method
            actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            print(f"Step {step_count}: Dino state after action: {actual_env.get_dino_state()}")
            
        state = preprocess_observation(state, device)

        total_reward += float(reward)
        
        # Debug reward changes
        if reward != 0.0:
            print(f"Step {step_count}: Reward changed to {reward}, Total: {total_reward}")
        
        if terminated:
            print(f"Game terminated at step {step_count} with final reward: {reward}")
            break

    print(f"Game ended after {step_count} steps")
    return total_reward


def ai_play(model_path: str):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Create the environment in normal mode for unlimited play
    # Use normal mode so the game only ends on collision, not time limit
    env = gym.make("Dino-v0", render_mode="human", game_mode="normal")
    env = Wrapper(env, k=4, image_size=(84, 84))  # Use same wrapper as training
    
    # Get dimensions for the model - must match training setup
    n_actions = env.action_space.n
    # The model was trained with 4 stacked frames
    n_observations = 4  # 4 stacked frames from wrapper
    
    # Create the model with the correct architecture
    policy_net = DQN(n_observations, n_actions).to(device)
    
    # Load the saved state dictionary
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    total_reward = play_with_model_wrapped(env, policy_net, device)

    print(f"Total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["human", "ai"])
    parser.add_argument("-m", "--model_path")

    args = parser.parse_args()
    if args.type == "human":
        human_play()
    else:
        ai_play(args.model_path)