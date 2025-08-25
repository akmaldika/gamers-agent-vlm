# Dungeon Game AI Agent

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --config-name <config_file_name>
```

## Available Configurations

### 1. Simple Navigation (Basic Movement)

```bash
# Straight corridor - 12 steps
python src/main.py --config-name straight_corridor

# Single turn navigation - 8 steps  
python src/main.py --config-name single_turn

# T-junction pathfinding - 20 steps
python src/main.py --config-name t_junction
```

### 2. Maze Exploration

```bash
# Simple maze - 30 steps
python src/main.py --config-name simple_maze

# Medium complexity maze - 50 steps
python src/main.py --config-name medium_maze

# Central player maze (symmetric) - 40 steps
python src/main.py --config-name central_player_maze
```

### 3. Single Threat Avoidance

```bash
# Monster in center of open room - 20 steps
python src/main.py --config-name center_monster_open_room

# Monster blocking left path - 25 steps
python src/main.py --config-name blocking_monster_left_path

# Monster in narrow corridor - 30 steps
python src/main.py --config-name narrow_corridor_monster
```

### 4. Multiple Threat Avoidance

```bash
# Grouped monsters - 35 steps
python src/main.py --config-name grouped_monsters

# Large room with three monsters - 40 steps
python src/main.py --config-name large_room_three_monsters

# Two paths with two monsters - 45 steps
python src/main.py --config-name two_path_two_monsters
```

### 5. Basic Combat

```bash
# Close combat start - 25 steps
python src/main.py --config-name close_combat_start

# Monster at junction - 30 steps
python src/main.py --config-name junction_monster

# Monster blocking stairs - 35 steps
python src/main.py --config-name monster_at_stairs
```

### 6. Tactical Combat Choice

```bash
# Approaching monster scenario - 40 steps
python src/main.py --config-name approaching_monster

# Corner monster flank maneuver - 45 steps
python src/main.py --config-name corner_monster_flank

# Strong monster: short vs long path - 50 steps
python src/main.py --config-name strong_monster_short_vs_long_path
```

### 7. Exploration with Threat

```bash
# Hidden stairs with patrol - 60 steps
python src/main.py --config-name hidden_stairs_patrol

# Large area with random monsters - 70 steps
python src/main.py --config-name large_random_monsters

# Multi-room exploration - 80 steps
python src/main.py --config-name multi_room_exploration
```

## Custom Configuration

You can also override individual parameters:

```bash
# Change model and settings
python src/main.py model=gpt-4o max_tokens=1000 temperature=0.3

# Use different map file
python src/main.py map_file="data/map_dataset/2_maze_exploration/simple_maze.txt"

# Combine config with overrides
python src/main.py --config-name simple_maze model=gpt-4o max_steps=100
```

## Configuration Structure

All configs include:
- **Map file**: Specific scenario map
- **Max steps**: Estimated steps needed for completion
- **Log naming**: Unique identifier for each scenario
- **Model settings**: AI model and parameters (configurable)