# 🤖 Robot Path Planning Simulator

A Python-based 2D robot simulator with LIDAR sensing, RRT* path planning, and multiple autonomous control algorithms.

[🎥 Watch Demo Video](demo.mp4)

## ✨ Features

- **360° LIDAR Sensor** - Configurable field-of-view and range
- **RRT* Path Planner** - Optimal collision-free path generation
- **Multiple Controllers** - Manual, State Machine, and Simple P-Controller
- **Real-time Visualization** - See sensor data, paths, and robot motion
- **Interactive** - Set goals and obstacles with mouse/keyboard

## 🚀 Quick Start

### Installation

```bash
pip install pygame numpy
```

### Run

```bash
python robot_simulator.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Arrow Keys** | Manual robot control |
| **G** | Generate obstacles |
| **M** | Toggle Random/Planning mode |
| **Space** | Change controller |
| **P** | Plan path (Planning mode) |
| **R** | Reset robot |
| **Click** | Set goal (Planning mode) |

## 🎯 Quick Tutorial

1. Press **G** to generate obstacles
2. Press **M** to switch to Planning mode
3. **Click** anywhere to set a goal
4. Press **P** to plan a path (yellow line appears)
5. Press **Space** to switch to Simple controller
6. Watch the robot navigate autonomously!

## 🔧 Configuration

Edit constants at the top of `robot_simulator.py`:

```python
OBSTACLE_DENSITY = 3     # Obstacle coverage (0-50%)
OBSTACLE_SIZE = 10       # Obstacle square size
LIDAR_OPACITY = 0.3      # Ray visibility (0.0-1.0)
```

### Controller Parameters

```python
'Kp': 0.05,              # Steering gain (path following)
'Ko': 0.02,              # Obstacle avoidance gain  
'Kv': 0.7,               # Velocity deceleration gain
'max_velocity': 1.5,     # Maximum speed
'waypoint_threshold': 25,# Waypoint advancement distance
```

## 📖 How It Works

### Path Planning (RRT*)
- Samples random points in free space
- Builds a tree of collision-free paths
- Rewires connections to optimize path cost
- Extracts shortest path from start to goal

### Path Following (Simple Controller)
- **Kp gain**: Steers toward current waypoint
- **Ko gain**: Steers away from LIDAR-detected obstacles
- **Kv gain**: Slows down when obstacles are nearby
- Automatically advances to next waypoint when close

### Collision Avoidance
- 360° LIDAR detects obstacles in all directions
- Proximity-weighted repulsion (closer = stronger)
- Combined with path following for smooth navigation
- Velocity control prevents high-speed collisions
<!-- 
## 🏗️ Code Structure

```
robot_simulator.py
├── Environment          # Obstacles, collision detection
├── LidarSensor         # Ray casting, detection
├── Robot               # Kinematics, state
├── RRTStar             # Path planning
├── Controllers         # Manual, StateMachine, Simple
└── RobotSimulator      # Main game loop -->
```

## 📚 Key Algorithms

- **RRT*** - Rapidly-exploring Random Tree Star (optimal path planning)
- **P-Controller** - Proportional control for trajectory tracking
- **Differential Drive** - Two-wheel robot kinematics
- **Ray Casting** - LIDAR obstacle detection

## 🛠️ Requirements

- Python 3.9+
- pygame >= 2.5.0
- numpy >= 1.24.0

## 📄 License

Educational project - free to use and modify.

---

**Built for learning path planning and autonomous navigation** 🚀