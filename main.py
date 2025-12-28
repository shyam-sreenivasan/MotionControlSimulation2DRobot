import pygame
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
ROBOT_RADIUS = 10
EDGE_BUFFER = 30
GOAL_RADIUS = 10
FPS = 60

# Colors
COLOR_BACKGROUND = (26, 26, 46)
COLOR_GRID = (42, 42, 62)
COLOR_OBSTACLE = (139, 69, 19)
COLOR_OBSTACLE_BORDER = (101, 67, 33)
COLOR_ROBOT = (0, 212, 255)
COLOR_ROBOT_CRASHED = (255, 107, 107)
COLOR_LIDAR_HIT = (255, 0, 0, 77)
COLOR_LIDAR_MISS = (0, 255, 0, 51)
COLOR_PATH = (255, 255, 0)
COLOR_GOAL = (255, 0, 0)
COLOR_GOAL_REACHED = (0, 255, 0)
COLOR_TREE = (100, 150, 255, 77)
COLOR_WHITE = (255, 255, 255)
COLOR_WAYPOINT_TARGET = (255, 136, 0)

OBSTACLE_DENSITY = 3  # Percentage of area covered by obstacles
OBSTACLE_SIZE = 10    # Size of each obstacle square
LIDAR_OPACITY = 0.3
# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float

@dataclass
class Detection:
    angle: float
    distance: float
    hit: bool

@dataclass
class RobotState:
    x: float
    y: float
    theta: float
    velocity: float

# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================
class Environment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
    
    def generate_obstacles(self, density: float, size: int) -> List[Obstacle]:
        obstacles = []
        grid_area = self.width * self.height
        target_obstacle_area = grid_area * (density / 100)
        obstacle_area = size * size
        num_obstacles = int(target_obstacle_area / obstacle_area)
        
        for _ in range(num_obstacles):
            x = random.random() * (self.width - size)
            y = random.random() * (self.height - size)
            obstacles.append(Obstacle(x, y, size, size))
        
        self.obstacles = obstacles
        return obstacles
    
    def is_in_bounds(self, x: float, y: float, radius: float) -> bool:
        return (x - radius >= 0 and x + radius <= self.width and
                y - radius >= 0 and y + radius <= self.height)
    
    def check_obstacle_collision(self, x: float, y: float, radius: float) -> bool:
        for obstacle in self.obstacles:
            if self._check_circle_rect_collision(x, y, radius, obstacle):
                return True
        return False
    
    def _check_circle_rect_collision(self, cx: float, cy: float, radius: float, rect: Obstacle) -> bool:
        closest_x = max(rect.x, min(cx, rect.x + rect.width))
        closest_y = max(rect.y, min(cy, rect.y + rect.height))
        
        distance_x = cx - closest_x
        distance_y = cy - closest_y
        distance_squared = distance_x * distance_x + distance_y * distance_y
        
        return distance_squared < radius * radius
    
    def get_random_safe_position(self, radius: float, edge_buffer: float) -> Tuple[float, float]:
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            x = edge_buffer + random.random() * (self.width - 2 * edge_buffer)
            y = edge_buffer + random.random() * (self.height - 2 * edge_buffer)
            
            if not self.check_obstacle_collision(x, y, radius):
                return x, y
            attempts += 1
        
        return self.width / 2, self.height / 2

# ============================================================================
# LIDAR SENSOR CLASS
# ============================================================================
class LidarSensor:
    def __init__(self, fov: float, range_: float, num_rays: int = 21):
        self.fov = fov
        self.range = range_
        self.num_rays = num_rays
    
    def scan(self, robot_x: float, robot_y: float, robot_theta: float, environment: Environment) -> List[Detection]:
        detections = []
        half_fov = (self.fov / 2) * math.pi / 180
        
        for i in range(self.num_rays):
            angle_offset = -half_fov + (i / (self.num_rays - 1)) * (2 * half_fov)
            ray_angle = robot_theta + angle_offset
            distance = self._cast_ray(robot_x, robot_y, ray_angle, environment)
            
            detections.append(Detection(
                angle=ray_angle,
                distance=distance,
                hit=distance < self.range
            ))
        
        return detections
    
    def _cast_ray(self, x: float, y: float, angle: float, environment: Environment) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        min_distance = float('inf')
        
        if dx < 0:
            min_distance = min(min_distance, -x / dx)
        if dx > 0:
            min_distance = min(min_distance, (environment.width - x) / dx)
        if dy < 0:
            min_distance = min(min_distance, -y / dy)
        if dy > 0:
            min_distance = min(min_distance, (environment.height - y) / dy)
        
        for obstacle in environment.obstacles:
            dist = self._ray_rect_intersection(x, y, dx, dy, obstacle)
            if dist is not None and dist > 0 and dist < min_distance:
                min_distance = dist
        
        return min_distance
    
    def _ray_rect_intersection(self, ray_x: float, ray_y: float, dir_x: float, dir_y: float, rect: Obstacle) -> Optional[float]:
        t_min = float('-inf')
        t_max = float('inf')
        
        if abs(dir_x) > 0.0001:
            t1 = (rect.x - ray_x) / dir_x
            t2 = (rect.x + rect.width - ray_x) / dir_x
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            if ray_x < rect.x or ray_x > rect.x + rect.width:
                return None
        
        if abs(dir_y) > 0.0001:
            t1 = (rect.y - ray_y) / dir_y
            t2 = (rect.y + rect.height - ray_y) / dir_y
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            if ray_y < rect.y or ray_y > rect.y + rect.height:
                return None
        
        if t_max >= t_min and t_max > 0:
            return t_min if t_min > 0 else t_max
        
        return None

# ============================================================================
# ROBOT CLASS
# ============================================================================
class Robot:
    def __init__(self, x: float, y: float, theta: float, radius: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = 0.0
        self.radius = radius
    
    def get_state(self) -> RobotState:
        return RobotState(self.x, self.y, self.theta, self.velocity)
    
    def set_state(self, x: float, y: float, theta: float, velocity: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity
    
    def apply_motion(self, velocity_cmd: float, theta_cmd: float) -> RobotState:
        self.velocity = velocity_cmd
        self.theta = ((theta_cmd % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi)
        
        new_x = self.x + self.velocity * math.cos(self.theta)
        new_y = self.y + self.velocity * math.sin(self.theta)
        
        return RobotState(new_x, new_y, self.theta, self.velocity)

# ============================================================================
# RRT* PATH PLANNER
# ============================================================================
@dataclass
class Node:
    x: float
    y: float
    cost: float = 0.0
    parent: Optional['Node'] = None

class RRTStar:
    def __init__(self, environment: Environment, start: Tuple[float, float], 
                 goal: Tuple[float, float], config: Dict):
        self.environment = environment
        self.start = start
        self.goal = goal
        self.config = {
            'max_iterations': config.get('max_iterations', 3000),
            'step_size': config.get('step_size', 20),
            'goal_bias': config.get('goal_bias', 0.1),
            'rewire_radius': config.get('rewire_radius', 50),
            'goal_threshold': config.get('goal_threshold', 25),
            'robot_radius': config.get('robot_radius', 15)
        }
        self.nodes: List[Node] = []
        self.edges: List[Tuple[Node, Node]] = []
    
    def plan(self) -> Optional[Dict]:
        self.nodes = [Node(self.start[0], self.start[1], 0.0, None)]
        self.edges = []
        
        for i in range(self.config['max_iterations']):
            sample = self._sample_point()
            nearest = self._find_nearest(sample)
            new_node = self._steer(nearest, sample)
            
            if not self._is_collision_free(nearest, new_node):
                continue
            
            nearby_nodes = self._find_nearby(new_node)
            best_parent = self._choose_best_parent(new_node, nearby_nodes)
            
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + self._distance(best_parent, new_node)
            
            self.nodes.append(new_node)
            self.edges.append((best_parent, new_node))
            
            self._rewire(new_node, nearby_nodes)
            
            if self._distance(new_node, Node(self.goal[0], self.goal[1])) < self.config['goal_threshold']:
                goal_node = Node(self.goal[0], self.goal[1], 
                               new_node.cost + self._distance(new_node, Node(self.goal[0], self.goal[1])),
                               new_node)
                self.nodes.append(goal_node)
                self.edges.append((new_node, goal_node))
                return {
                    'path': self._extract_path(goal_node),
                    'tree': {'nodes': self.nodes, 'edges': self.edges}
                }
        
        return None
    
    def _sample_point(self) -> Node:
        if random.random() < self.config['goal_bias']:
            return Node(self.goal[0], self.goal[1])
        return Node(
            random.random() * self.environment.width,
            random.random() * self.environment.height
        )
    
    def _find_nearest(self, point: Node) -> Node:
        min_dist = float('inf')
        nearest = None
        for node in self.nodes:
            d = self._distance(node, point)
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest
    
    def _steer(self, from_node: Node, to_node: Node) -> Node:
        d = self._distance(from_node, to_node)
        if d < self.config['step_size']:
            return Node(to_node.x, to_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        return Node(
            from_node.x + self.config['step_size'] * math.cos(theta),
            from_node.y + self.config['step_size'] * math.sin(theta)
        )
    
    def _is_collision_free(self, from_node: Node, to_node: Node) -> bool:
        steps = int(math.ceil(self._distance(from_node, to_node) / 5))
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            if not self.environment.is_in_bounds(x, y, self.config['robot_radius']):
                return False
            if self.environment.check_obstacle_collision(x, y, self.config['robot_radius']):
                return False
        return True
    
    def _find_nearby(self, node: Node) -> List[Node]:
        nearby = []
        for other in self.nodes:
            if self._distance(node, other) < self.config['rewire_radius']:
                nearby.append(other)
        return nearby
    
    def _choose_best_parent(self, node: Node, candidates: List[Node]) -> Node:
        best_parent = candidates[0]
        best_cost = best_parent.cost + self._distance(best_parent, node)
        
        for candidate in candidates:
            cost = candidate.cost + self._distance(candidate, node)
            if cost < best_cost and self._is_collision_free(candidate, node):
                best_cost = cost
                best_parent = candidate
        return best_parent
    
    def _rewire(self, new_node: Node, nearby_nodes: List[Node]):
        for nearby in nearby_nodes:
            new_cost = new_node.cost + self._distance(new_node, nearby)
            if new_cost < nearby.cost and self._is_collision_free(new_node, nearby):
                nearby.parent = new_node
                nearby.cost = new_cost
    
    def _extract_path(self, goal_node: Node) -> List[Dict]:
        path = []
        current = goal_node
        while current is not None:
            path.insert(0, {'x': current.x, 'y': current.y, 'theta': 0.0})
            current = current.parent
        
        for i in range(len(path)):
            if i < len(path) - 1:
                dx = path[i + 1]['x'] - path[i]['x']
                dy = path[i + 1]['y'] - path[i]['y']
                path[i]['theta'] = math.atan2(dy, dx)
            elif i > 0:
                path[i]['theta'] = path[i - 1]['theta']
        
        return path
    
    def _distance(self, a: Node, b: Node) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy)

# ============================================================================
# CONTROLLER BASE CLASS
# ============================================================================
class Controller:
    def __init__(self, name: str):
        self.name = name
    
    def compute_control(self, robot: Robot, sensor: LidarSensor, 
                       environment: Environment, trajectory: Optional[List]) -> Dict:
        raise NotImplementedError("compute_control must be implemented by subclass")

# ============================================================================
# MANUAL CONTROLLER
# ============================================================================
class ManualController(Controller):
    def __init__(self, max_velocity: float, acceleration: float, rotation_speed: float):
        super().__init__("Manual")
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.rotation_speed = rotation_speed
        self.keys = {}
    
    def set_keys(self, keys: Dict):
        self.keys = keys
    
    def compute_control(self, robot: Robot, sensor: LidarSensor, 
                       environment: Environment, trajectory: Optional[List]) -> Dict:
        state = robot.get_state()
        new_velocity = state.velocity
        new_theta = state.theta
        
        detections = sensor.scan(state.x, state.y, state.theta, environment)
        forward_safe = self._is_forward_motion_safe(detections, state.velocity)
        
        if self.keys.get(pygame.K_UP):
            if forward_safe:
                new_velocity = min(new_velocity + self.acceleration, self.max_velocity)
            else:
                new_velocity = max(new_velocity - self.acceleration * 2, 0)
        if self.keys.get(pygame.K_DOWN):
            new_velocity = max(new_velocity - self.acceleration, -self.max_velocity)
        
        if not self.keys.get(pygame.K_UP) and not self.keys.get(pygame.K_DOWN):
            if abs(new_velocity) < 0.1:
                new_velocity = 0
            else:
                new_velocity *= 0.98
        
        if self.keys.get(pygame.K_LEFT):
            new_theta -= self.rotation_speed
        if self.keys.get(pygame.K_RIGHT):
            new_theta += self.rotation_speed
        
        return {'velocity': new_velocity, 'theta': new_theta}
    
    def _is_forward_motion_safe(self, detections: List[Detection], velocity: float) -> bool:
        stopping_distance = (abs(velocity) * abs(velocity)) / (2 * self.acceleration * 2) + 30
        
        for detection in detections:
            if detection.hit and detection.distance < stopping_distance:
                return False
        return True

# ============================================================================
# STATE MACHINE CONTROLLER
# ============================================================================
class StateMachineController(Controller):
    def __init__(self, max_velocity: float, acceleration: float):
        super().__init__("StateMachine")
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.linear_motion_enabled = True
        self.is_rotating = False
        self.target_angle = 0.0
        self.rotated_so_far = 0.0
        self.direction = 0
    
    def compute_control(self, robot: Robot, sensor: LidarSensor, 
                       environment: Environment, trajectory: Optional[List]) -> Dict:
        state = robot.get_state()
        detections = sensor.scan(state.x, state.y, state.theta, environment)
        forward_safe = self._is_forward_motion_safe(detections, state.velocity)
        
        new_velocity = state.velocity
        new_theta = state.theta
        
        if not self.linear_motion_enabled:
            new_velocity = 0
            rotation_speed = 5 * math.pi / 180
            new_theta += self.direction * rotation_speed
            self.rotated_so_far += abs(rotation_speed)
            
            if self.rotated_so_far >= self.target_angle:
                test_detections = sensor.scan(state.x, state.y, new_theta, environment)
                if self._is_forward_motion_safe(test_detections, 0):
                    self.linear_motion_enabled = True
                    self.is_rotating = False
                else:
                    self.target_angle += (10 * math.pi / 180)
        else:
            if not forward_safe and abs(state.velocity) < 0.1:
                self.linear_motion_enabled = False
                self.is_rotating = True
                self.direction = -1 if random.random() < 0.5 else 1
                self.target_angle = 90 * math.pi / 180
                self.rotated_so_far = 0
                new_velocity = 0
            elif forward_safe:
                new_velocity = min(new_velocity + self.acceleration, self.max_velocity)
            else:
                new_velocity = max(new_velocity - self.acceleration * 2, 0)
        
        return {'velocity': new_velocity, 'theta': new_theta}
    
    def _is_forward_motion_safe(self, detections: List[Detection], velocity: float) -> bool:
        stopping_distance = (abs(velocity) * abs(velocity)) / (2 * self.acceleration * 2) + 30
        
        for detection in detections:
            if detection.hit and detection.distance < stopping_distance:
                return False
        return True
    
    def reset(self):
        self.linear_motion_enabled = True
        self.is_rotating = False
        self.target_angle = 0.0
        self.rotated_so_far = 0.0
        self.direction = 0

# ============================================================================
# SIMPLE PATH FOLLOWING CONTROLLER
# ============================================================================
class SimplePathFollowingController(Controller):
    def __init__(self, config: Dict):
        super().__init__("SimplePathFollowing")
        self.config = config
        self.trajectory = None
        self.current_waypoint_index = 0
        self.waypoint_threshold = config.get('waypoint_threshold', 25)
        self.Kp = config.get('Kp', 1)           # Steering gain
        self.Ko = config.get('Ko', 10)           # Obstacle avoidance gain
        self.Kv = config.get('Kv', 1)           # NEW: Velocity/deceleration gain
        self.obstacle_threshold = config.get('obstacle_threshold', 100)  # Distance threshold
        self.min_velocity = config.get('min_velocity', 0.3)  # Minimum velocity
    
    def set_trajectory(self, trajectory: List):
        self.trajectory = trajectory
        self.current_waypoint_index = 0
    
    def compute_control(self, robot: Robot, sensor: LidarSensor, 
                       environment: Environment, trajectory: Optional[List]) -> Dict:
        if not self.trajectory or len(self.trajectory) == 0:
            return {'velocity': 0, 'theta': robot.get_state().theta}
        
        state = robot.get_state()
        
        if self.current_waypoint_index >= len(self.trajectory):
            return {'velocity': 0, 'theta': state.theta}
        
        # Find closest waypoint ahead
        best_index = self.current_waypoint_index
        min_dist = float('inf')
        
        for i in range(self.current_waypoint_index, len(self.trajectory)):
            wp = self.trajectory[i]
            dist = math.sqrt((wp['x'] - state.x) ** 2 + (wp['y'] - state.y) ** 2)
            
            if dist < min_dist:
                min_dist = dist
                best_index = i
            
            if i > self.current_waypoint_index and dist > min_dist + 50:
                break
        
        self.current_waypoint_index = best_index
        target = self.trajectory[self.current_waypoint_index]
        
        dist_to_target = math.sqrt(
            (target['x'] - state.x) ** 2 + (target['y'] - state.y) ** 2
        )
        
        if dist_to_target < self.waypoint_threshold and self.current_waypoint_index < len(self.trajectory) - 1:
            self.current_waypoint_index += 1
            return self.compute_control(robot, sensor, environment, trajectory)
        
        target_angle = math.atan2(target['y'] - state.y, target['x'] - state.x)
        
        heading_error = target_angle - state.theta
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Get LIDAR detections
        detections = sensor.scan(state.x, state.y, state.theta, environment)
        
        # ========================================================================
        # NEW: Calculate velocity based on obstacle proximity
        # ========================================================================
        min_obstacle_distance = float('inf')
        obstacle_avoidance = 0
        
        for detection in detections:
            if detection.hit:
                # Track minimum obstacle distance
                min_obstacle_distance = min(min_obstacle_distance, detection.distance)
                
                # Obstacle avoidance steering (existing code)
                if detection.distance < sensor.range * 0.5:
                    angle_diff = detection.angle - state.theta
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    proximity_weight = 1.0 - (detection.distance / (sensor.range * 0.5))
                    obstacle_avoidance -= (1 if angle_diff > 0 else -1) * proximity_weight * self.Ko
        
        # ========================================================================
        # VELOCITY CONTROL based on obstacle distance
        # ========================================================================
        if min_obstacle_distance < self.obstacle_threshold:
            # Linear deceleration: closer obstacle = slower speed
            # velocity_scale ranges from 0 to 1 based on distance
            velocity_scale = (min_obstacle_distance / self.obstacle_threshold)
            
            # Apply Kv gain to control how aggressively we slow down
            # Higher Kv = more aggressive slowing
            adjusted_scale = 1.0 - (self.Kv * (1.0 - velocity_scale))
            adjusted_scale = max(0, min(1, adjusted_scale))  # Clamp to [0, 1]
            
            # Calculate velocity with minimum speed floor
            velocity = self.config['max_velocity'] * adjusted_scale
            velocity = max(self.min_velocity, velocity)
        else:
            # No obstacles nearby - full speed ahead!
            velocity = self.config['max_velocity']
        
        # Combine steering commands
        steering = self.Kp * heading_error + obstacle_avoidance
        clamped_steering = max(-0.15, min(0.15, steering))
        new_theta = state.theta + clamped_steering
        
        return {'velocity': velocity, 'theta': new_theta}
    
    def reset(self):
        self.current_waypoint_index = 0
    
    def get_current_waypoint_index(self) -> int:
        return self.current_waypoint_index
# ============================================================================
# MAIN APPLICATION
# ============================================================================
class RobotSimulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Robot Simulator - Python Version")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        
        # Initialize components
        self.environment = Environment(WIDTH, HEIGHT)
        self.robot = Robot(WIDTH / 2, HEIGHT / 2, 0, ROBOT_RADIUS)
        self.sensor = LidarSensor(360, 350)
        
        # Controllers
        self.manual_controller = ManualController(1.5, 0.15, 0.05)
        self.state_machine_controller = StateMachineController(1.5, 0.15)
        self.simple_controller = SimplePathFollowingController({
            'Kp': 0.05,
            'Ko': 0.02,
            'max_velocity': 1.5,
            'waypoint_threshold': 25
        })
        
        # State
        self.mode = 'random'  # 'random' or 'planning'
        self.control_mode = 'manual'  # 'manual', 'automatic', 'simple'
        self.obstacles = []
        self.goal = None
        self.trajectory = None
        self.rrt_tree = None
        self.lidar_detections = []
        self.crashed = False
        self.goal_reached = False
        self.keys = {}
        self.lidar_opacity = 0.3  # ← NEW: Default opacity (0.0 to 1.0)

        # UI State
        self.adding_goal = False
        self.show_tree = True
        
        self.running = True
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    self.keys[event.key] = True
                    self.manual_controller.set_keys(self.keys)
                elif event.key == pygame.K_g:  # Generate obstacles
                    self.generate_obstacles()
                elif event.key == pygame.K_m:  # Toggle mode
                    self.mode = 'planning' if self.mode == 'random' else 'random'
                elif event.key == pygame.K_SPACE:  # Toggle control mode
                    if self.mode == 'random':
                        self.control_mode = 'automatic' if self.control_mode == 'manual' else 'manual'
                    else:
                        if self.trajectory:
                            modes = ['manual', 'simple']
                            idx = modes.index(self.control_mode)
                            self.control_mode = modes[(idx + 1) % len(modes)]
                elif event.key == pygame.K_r:  # Reset
                    self.reset_robot()
                elif event.key == pygame.K_p:  # Plan path
                    if self.goal and self.mode == 'planning':
                        self.plan_path()
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    self.keys[event.key] = False
                    self.manual_controller.set_keys(self.keys)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.mode == 'planning':
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if event.button == 1:  # Left click - set goal
                        self.goal = (mouse_x, mouse_y)
                        self.goal_reached = False
    
    def generate_obstacles(self):
        self.obstacles = self.environment.generate_obstacles(OBSTACLE_DENSITY, OBSTACLE_SIZE)
        x, y = self.environment.get_random_safe_position(ROBOT_RADIUS, EDGE_BUFFER)
        self.robot.set_state(x, y, random.random() * 2 * math.pi, 0)
        self.crashed = False
    
    def plan_path(self):
        state = self.robot.get_state()
        rrt = RRTStar(
            self.environment,
            (state.x, state.y),
            self.goal,
            {
                'max_iterations': 3000,
                'step_size': 30,
                'goal_bias': 0.1,
                'rewire_radius': 50,
                'goal_threshold': 25,
                'robot_radius': ROBOT_RADIUS
            }
        )
        
        result = rrt.plan()
        if result:
            self.trajectory = result['path']
            self.rrt_tree = result['tree']
            self.simple_controller.set_trajectory(self.trajectory)
        else:
            print("No path found!")
    
    def reset_robot(self):
        if self.mode == 'planning' and self.trajectory:
            start_point = self.trajectory[0]
            self.robot.set_state(start_point['x'], start_point['y'], start_point['theta'], 0)
        else:
            x, y = self.environment.get_random_safe_position(ROBOT_RADIUS, EDGE_BUFFER)
            self.robot.set_state(x, y, random.random() * 2 * math.pi, 0)
        
        self.crashed = False
        self.goal_reached = False
        self.state_machine_controller.reset()
        self.simple_controller.reset()
    
    def update(self):
        if self.crashed or self.goal_reached:
            return
        
        state = self.robot.get_state()
        self.lidar_detections = self.sensor.scan(state.x, state.y, state.theta, self.environment)
        
        # Select controller
        if self.mode == 'random':
            controller = self.manual_controller if self.control_mode == 'manual' else self.state_machine_controller
        else:
            controller = self.manual_controller if self.control_mode == 'manual' else self.simple_controller
        
        control = controller.compute_control(self.robot, self.sensor, self.environment, self.trajectory)
        predicted_state = self.robot.apply_motion(control['velocity'], control['theta'])
        
        # Check collisions
        if not self.environment.is_in_bounds(predicted_state.x, predicted_state.y, ROBOT_RADIUS):
            self.crashed = True
            return
        
        if self.environment.check_obstacle_collision(predicted_state.x, predicted_state.y, ROBOT_RADIUS):
            self.crashed = True
            return
        
        self.robot.set_state(predicted_state.x, predicted_state.y, predicted_state.theta, predicted_state.velocity)
        
        # Check goal reached
        if self.mode == 'planning' and self.goal:
            dist = math.sqrt((predicted_state.x - self.goal[0]) ** 2 + (predicted_state.y - self.goal[1]) ** 2)
            if dist < 25:
                self.goal_reached = True
    
    def draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        
        # Draw grid
        for i in range(50, WIDTH, 50):
            pygame.draw.line(self.screen, COLOR_GRID, (i, 0), (i, HEIGHT))
        for i in range(50, HEIGHT, 50):
            pygame.draw.line(self.screen, COLOR_GRID, (0, i), (WIDTH, i))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, COLOR_OBSTACLE, 
                        (obstacle.x, obstacle.y, obstacle.width, obstacle.height))
            pygame.draw.rect(self.screen, COLOR_OBSTACLE_BORDER, 
                        (obstacle.x, obstacle.y, obstacle.width, obstacle.height), 2)
        
        # Draw RRT tree
        if self.mode == 'planning' and self.show_tree and self.rrt_tree:
            for edge in self.rrt_tree['edges']:
                pygame.draw.line(self.screen, COLOR_TREE, 
                            (edge[0].x, edge[0].y), (edge[1].x, edge[1].y), 1)
        
        # Draw trajectory
        if self.mode == 'planning' and self.trajectory:
            points = [(wp['x'], wp['y']) for wp in self.trajectory]
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLOR_PATH, False, points, 3)
            
            for wp in self.trajectory:
                pygame.draw.circle(self.screen, COLOR_PATH, (int(wp['x']), int(wp['y'])), 4)
            
            # Draw current waypoint target
            if self.control_mode == 'simple':
                idx = self.simple_controller.get_current_waypoint_index()
                if idx < len(self.trajectory):
                    target = self.trajectory[idx]
                    pygame.draw.circle(self.screen, COLOR_WAYPOINT_TARGET, 
                                    (int(target['x']), int(target['y'])), 10, 2)
        
        # Draw goal
        if self.mode == 'planning' and self.goal:
            color = COLOR_GOAL_REACHED if self.goal_reached else COLOR_GOAL
            pygame.draw.circle(self.screen, color, (int(self.goal[0]), int(self.goal[1])), GOAL_RADIUS, 3)
            pygame.draw.line(self.screen, color, 
                        (self.goal[0] - GOAL_RADIUS - 5, self.goal[1]), 
                        (self.goal[0] + GOAL_RADIUS + 5, self.goal[1]), 3)
            pygame.draw.line(self.screen, color, 
                        (self.goal[0], self.goal[1] - GOAL_RADIUS - 5), 
                        (self.goal[0], self.goal[1] + GOAL_RADIUS + 5), 3)
        
        # ========================================================================
        # Draw LIDAR rays with configurable opacity
        # ========================================================================
        if LIDAR_OPACITY > 0:  # Only draw if opacity > 0
            state = self.robot.get_state()
            
            # Create temporary surface for transparency
            ray_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            
            for detection in self.lidar_detections:
                end_x = state.x + detection.distance * math.cos(detection.angle)
                end_y = state.y + detection.distance * math.sin(detection.angle)
                
                # Apply opacity to colors
                if detection.hit:
                    # Red ray with opacity
                    color = (255, 0, 0, int(77 * LIDAR_OPACITY))
                else:
                    # Green ray with opacity
                    color = (0, 255, 0, int(51 * LIDAR_OPACITY))
                
                pygame.draw.line(ray_surface, color, (state.x, state.y), (end_x, end_y), 1)
                
                # Draw hit points with opacity
                if detection.hit:
                    hit_color = (255, 0, 0, int(153 * LIDAR_OPACITY))
                    pygame.draw.circle(ray_surface, hit_color, (int(end_x), int(end_y)), 2)
            
            # Blit the transparent surface onto the main screen
            self.screen.blit(ray_surface, (0, 0))
        # ========================================================================
        
            # Draw robot
            state = self.robot.get_state()
            color = COLOR_ROBOT_CRASHED if self.crashed else COLOR_ROBOT
            pygame.draw.circle(self.screen, color, (int(state.x), int(state.y)), ROBOT_RADIUS)
            
            # Draw robot direction
            end_x = state.x + ROBOT_RADIUS * math.cos(state.theta)
            end_y = state.y + ROBOT_RADIUS * math.sin(state.theta)
            pygame.draw.line(self.screen, COLOR_WHITE, (state.x, state.y), (end_x, end_y), 3)
            pygame.draw.circle(self.screen, COLOR_WHITE, (int(end_x * 0.6 + state.x * 0.4), 
                                                        int(end_y * 0.6 + state.y * 0.4)), 3)
            
            # Draw UI text
            mode_text = f"Mode: {'Path Planning' if self.mode == 'planning' else 'Random Driving'}"
            text_surface = self.font.render(mode_text, True, COLOR_WHITE)
            self.screen.blit(text_surface, (10, 10))
            
            controller_text = f"Controller: {self.control_mode}"
            text_surface = self.font.render(controller_text, True, COLOR_WHITE)
            self.screen.blit(text_surface, (10, 30))
            
            pos_text = f"Position: ({int(state.x)}, {int(state.y)})"
            text_surface = self.font.render(pos_text, True, COLOR_WHITE)
            self.screen.blit(text_surface, (10, 50))
            
            if self.mode == 'planning' and self.goal:
                dist = math.sqrt((state.x - self.goal[0]) ** 2 + (state.y - self.goal[1]) ** 2)
                dist_text = f"Distance to Goal: {int(dist)}"
                text_surface = self.font.render(dist_text, True, COLOR_WHITE)
                self.screen.blit(text_surface, (10, 70))
            
            # Draw status messages
            if self.crashed:
                text = self.font.render("CRASHED! Press R to reset", True, (255, 0, 0))
                self.screen.blit(text, (WIDTH // 2 - 100, HEIGHT // 2))
            elif self.goal_reached:
                text = self.font.render("GOAL REACHED! Press R to reset", True, (0, 255, 0))
                self.screen.blit(text, (WIDTH // 2 - 120, HEIGHT // 2))
            
            # Draw controls help
            help_y = HEIGHT - 100
            help_texts = [
                "Controls: Arrow keys = Move | G = Generate obstacles | M = Toggle mode",
                "Space = Change controller | R = Reset | P = Plan path | Click = Set goal"
            ]
            for i, text in enumerate(help_texts):
                text_surface = self.font.render(text, True, (150, 150, 150))
                self.screen.blit(text_surface, (10, help_y + i * 20))
            
            pygame.display.flip()
        
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    app = RobotSimulator()
    app.run()