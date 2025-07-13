import argparse
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import override
from tqdm import tqdm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


# --- Parameters ---
SPACE_SIZE = (20, 20)
MAX_VELOCITY_PER_FRAME = 0.5
AGENT_RADIUS_SCALE = 0.8
GROUP_POSITIONS = [[18, 18], [2, 2], [2, 18], [18, 2]]
GROUP_GOALS = [[2, 2], [18, 18], [18, 2], [2, 18]]
NEIGHBOR_RADIUS = 1
ALIGN_WEIGHT = 10
SEPARATE_WEIGHT = 30
COHESION_WEIGHT = 10
OBSTACLES_SEPARATE_WEIGHT = 70
STEER_FORCE_MULTIPLIER = 10.0
EPSILON = 1e-8
OBSTACLE_RADIUS = 1
OBSTACLES = np.array([[10, 12], [10, 8], [15, 15], [5, 5]])


# --- Parse CLI Arguments ---
parser = argparse.ArgumentParser(description="Crowd simulation with marker-based navigation.")
parser.add_argument('--use-obstacles', action='store_true', help='Enable position validation for markers')
parser.add_argument('--allow-biocrowds-collision', action='store_true', help='Enable collision avoidance for biocrowds')
parser.add_argument('--radius', type=float, default=0.3, help='Radius of each agent')
parser.add_argument('--perception', type=float, default=1.2, help='Perception radius')
parser.add_argument('--num-agents', type=int, default=60, help='Total number of agents')
parser.add_argument('--num-groups', type=int, default=2, help='Total number of groups')
parser.add_argument('--marker-density', type=int, default=15, help='Total number of markers per tile')
parser.add_argument('--agent-types', type=str, default='biocrowds,boids', help='Total number of markers per tile')
args = parser.parse_args()


# --- Classes ---
class Marker(object):
    def __init__(self, position):
        self.position = np.array(position)
        self.closer_agent = None
        self.closer_agent_distance = float('inf')

class Agent(ABC):
    def __init__(self, position, goal):
      scale = np.random.uniform(-1, 1, size=2)
      self.position = np.array(position + scale, dtype=float)
      self.goal = goal
      self.velocity = np.zeros(2)
      self.max_velocity = MAX_VELOCITY_PER_FRAME
    
    def clip_velocity(self, velocity):
        clip_percentage = min(self.max_velocity / (np.linalg.norm(velocity) + EPSILON), 1)
        return velocity * clip_percentage

    @abstractmethod
    def update(self, agents: np.ndarray):
        """Update the agent's position based on its behavior."""
        pass

class BioCrowdsAgent(Agent):
    def __init__(self, position, goal):
        super().__init__(position, goal)
        self.close_markers = []
    
    @override
    def update(self, agents: np.ndarray):
      agent_markers_positions = np.array([marker.position for marker in self.close_markers if marker.closer_agent is self], dtype=float)

      if len(agent_markers_positions) == 0:
          # No markers assigned to this agent, sit still
          move = np.zeros(2)
      else:
          to_goal = self.goal - self.position
          to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-5)

          # Implement weighted vector sum over agent markers
          vectors = agent_markers_positions - self.position
          norms = np.linalg.norm(vectors, axis=1, keepdims=True)
          norms[norms < 1e-5] = 1e-5
          cosines = ((vectors / norms) @ to_goal_norm.T)
          weights = (1 + cosines) / (1 + norms.flatten())
          weighted_sum = (vectors * weights[:, None]).sum(axis=0)

          if np.linalg.norm(weighted_sum) > 0:
              speed_vector = weighted_sum
          else:
              # In case the weighted sum is zero, use the direction to the goal
              # This is like cheating, since we should only move based on markers calculations
              speed_vector = to_goal_norm

          # Calculated move vector (x,y), clipped to max velocity
          calculated_move = self.clip_velocity(speed_vector)
          move = self.__calculate_move_with_collision_avoidance(agents, calculated_move)
         
      
      # Apply the move vector to the agent's position
      self.close_markers = []
      self.velocity = move
      self.position += self.velocity

    def __calculate_move_with_collision_avoidance(self, agents, calculated_move):
      if args.allow_biocrowds_collision:
          return calculated_move

      # Apply collision avoidance with other agents
      other_agents_positions = np.array([agent.position for agent in agents if agent is not self], dtype=float)

      # Calculate the minimum distance to other agents
      # and the new minimum distance after applying the calculated move
      min_agent_distance = min(np.linalg.norm(other_agents_positions - self.position, axis=1))
      new_min_agent_distance = min(np.linalg.norm(other_agents_positions - (self.position + calculated_move), axis=1))

      # We are currently collided
      if min_agent_distance < 2 * args.radius:
          # If we are moving away from collison, allow full movement
          return calculated_move if new_min_agent_distance > min_agent_distance else np.zeros(2)
      
      # We are not collided. Calculate the percentage of movement we can do
      total_movement = np.linalg.norm(calculated_move)
      allowed_percetage = 1.0
      if new_min_agent_distance < 2 * args.radius:
        overlap_distance = 2 * args.radius - new_min_agent_distance
        allowed_percetage = (total_movement - overlap_distance) / total_movement
      
      return calculated_move * allowed_percetage

class BoidsAgent(Agent):
    def __init__(self, position, goal):
        super().__init__(position, goal)
        self.acceleration = np.zeros(2)

    @override
    def update(self, agents: np.ndarray):
        near_neighbors = self.__get_near_agents(agents)

        self.__steer(self.goal)
        self.__align(near_neighbors)
        self.__separate(near_neighbors)
        self.__cohesion(near_neighbors)
        self.__avoid_obstacles(OBSTACLES, self.goal)

        self.__move()
    
    def __get_near_agents(self, agents):
        return [agent for agent in agents if np.linalg.norm(agent.position - self.position) < NEIGHBOR_RADIUS and agent is not self]

    def __move(self):
        self.velocity = self.clip_velocity(self.velocity + self.acceleration)
        self.position += self.velocity
        self.acceleration = np.zeros(2)

    def __apply_force(self, force_vector):
        self.acceleration += force_vector

    def __steer(self, goal):
        to_goal = goal - self.position
        steer_vector = (to_goal - self.velocity)
        
        # We apply some gain to make sure agents will not slow down until they reach the goal
        self.__apply_force(steer_vector)

    def __align(self, agents):
        if agents:
            avg_velocity = np.mean([agent.velocity for agent in agents], axis=0)
            self.__apply_force((avg_velocity - self.velocity) * ALIGN_WEIGHT)

    def __separate(self, agents):
        if agents:
            separation_force = np.zeros(2)
            for agent in agents:
                offset = self.position - agent.position
                dist = np.linalg.norm(offset)
                separation_force += offset / dist
            
            self.__apply_force((separation_force - self.velocity) * SEPARATE_WEIGHT)

    def __cohesion(self, agents):
        if agents:
            avg_position = np.mean([agent.position for agent in agents], axis=0)
            self.__apply_force((avg_position - self.position) * COHESION_WEIGHT)

    def __avoid_obstacles(self, obstacles, goal):
        if not args.use_obstacles: return
        for obstacle in obstacles:
            to_agent = self.position - obstacle
            dist = np.linalg.norm(to_agent)
            if dist < NEIGHBOR_RADIUS * 2:
                to_agent_norm = to_agent / (dist + EPSILON)

                # Direction from obstacle to goal
                to_goal = goal - obstacle
                targeted_avoidance = to_goal - to_agent

                # Generate two perpendicular vectors to to_agent
                perp1 = np.array([-to_agent_norm[1], to_agent_norm[0]])
                perp2 = np.array([ to_agent_norm[1], -to_agent_norm[0]])

                # Choose the one better aligned with the intended escape
                dot1 = np.dot(perp1, targeted_avoidance)
                dot2 = np.dot(perp2, targeted_avoidance)
                best_perp = perp1 if dot1 > dot2 else perp2

                # Add a small amount of this directional noise
                noise = best_perp * 1.5

                # Final avoidance vector
                avoidance_direction = to_agent_norm + self.velocity + noise
                avoidance_force = avoidance_direction / (np.linalg.norm(avoidance_direction) + EPSILON)

                self.__apply_force(avoidance_force * OBSTACLES_SEPARATE_WEIGHT)


# --- Helper Variables and Functions ---
# Get a list with Class references of available agent types, from argparser
agent_class_map = {
    'biocrowds': BioCrowdsAgent,
    'boids': BoidsAgent
}
available_agents = [agent_class_map[agent_type.strip()] for agent_type in args.agent_types.split(',')]

def valid_position(position):
    if not args.use_obstacles:
        return True
    for obstacle in OBSTACLES:
        if np.linalg.norm(position - obstacle) < OBSTACLE_RADIUS * max(1.1, 0.9 + args.radius):
            return False
    return True

def generate_markers(space_size, marker_density):
    markers = []
    min_distance = 3 / marker_density  # Minimum distance between markers in each tile

    for x in range(space_size[0]):
        for y in range(space_size[1]):
            tile_markers = []
            attempts = 0
            max_attempts = marker_density * 10  # prevent infinite loops

            while len(tile_markers) < marker_density and attempts < max_attempts:
                # Generate a random (x, y) inside this tile
                rx = np.random.uniform(x, x + 1)
                ry = np.random.uniform(y, y + 1)
                candidate = np.array([rx, ry])

                # Check if candidate is far enough from existing markers in this tile
                if all(np.linalg.norm(candidate - m.position) >= min_distance for m in tile_markers):
                    if valid_position(candidate):
                        tile_markers.append(Marker(candidate))

                attempts += 1

            markers.extend(tile_markers)

    return np.array(markers)

def get_agent_class(idx, position, goal):
    global available_agents

    # Make the agent choice based on the index and available agents
    choice = idx % len(available_agents)
    return available_agents[choice](position, goal)

# --- Create Markers ---
num_markers = int(SPACE_SIZE[0] * SPACE_SIZE[1] * args.marker_density)
markers = generate_markers(SPACE_SIZE, args.marker_density)
marker_positions = np.array([marker.position for marker in markers])


# --- Create Agents and Goals ---
group_size = args.num_agents // args.num_groups
agents = []
goals = []
for i in range(args.num_groups):
    pos = GROUP_POSITIONS[i]
    goal = GROUP_GOALS[i]

    agents.extend([get_agent_class(i, pos, goal) for _ in range(group_size)])
    goals.extend([goal] * group_size)

agents = np.array(agents)
goals = np.array(goals)

# --- Plot Setup ---
fig, ax = plt.subplots()
colors = ['red', 'blue', 'green', 'purple']
legend_items = [Line2D([0], [0], marker='o', color='w', label=available_agents[i%len(available_agents)].__name__,
                       markerfacecolor=colors[i], markersize=8)
                for i in range(args.num_groups)]


# --- Simulation Loop ---
total_steps = int(50 / MAX_VELOCITY_PER_FRAME)
for step in tqdm(range(total_steps), desc="Simulating"):
    plt.cla()
    ax.set_xlim(0, SPACE_SIZE[0])
    ax.set_ylim(0, SPACE_SIZE[1])
    ax.set_title("Crowd Simulation | Step: {}/{}".format(step + 1, total_steps))
    ax.legend(handles=legend_items, loc='upper center', title='Agent Types', fontsize=8)

    # Plot markers
    ax.scatter(marker_positions[:, 0], marker_positions[:, 1], color='gray', s=1)

    # Plot goals
    for g in range(args.num_groups):
        ax.scatter(goals[g * group_size:(g + 1) * group_size, 0],
                   goals[g * group_size:(g + 1) * group_size, 1],
                   color=colors[g], marker='x')

    # Plot obstacles
    if args.use_obstacles:
      for obstacle in OBSTACLES:
        ax.add_patch(Circle(obstacle, OBSTACLE_RADIUS, color='gray', alpha=0.4))

    # Assign closest agent to each marker
    for marker in markers:
        marker.closer_agent = None
        marker.closer_agent_distance = float('inf')

        for agent in agents:
            distance = np.linalg.norm(marker.position - agent.position)
            
            # We should only consider agents within the perception radius
            if distance < args.perception and marker.closer_agent_distance > distance:
                marker.closer_agent = agent
                marker.closer_agent_distance = distance
                
                # For BioCrowds agents, we keep track of the markers they are close
                # This improves performance by avoiding repeated distance calculations during update()
                if isinstance(agent, BioCrowdsAgent):
                  agent.close_markers.append(marker)


    # Move agents
    for idx, (agent, goal) in enumerate(zip(agents, goals)):
        agent.update(agents)
        group = idx // group_size

        ax.scatter(agent.position[0], agent.position[1], color=colors[group], s=5)
        ax.add_patch(Circle(agent.position, args.radius * 0.8, color=colors[group], fill=True, linestyle='-', alpha=0.5))
        ax.add_patch(Circle(agent.position, args.perception, color=colors[group], fill=False, linestyle='--', alpha=0.5))

    plt.pause(0.001)

plt.show()
