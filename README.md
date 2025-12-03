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

# Medium maze set (category configs)
python src/main.py --config-dir conf/02_maze_exploration --config-name medium_maze_1
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

### System Prompt Caching

The default configuration now caches the system prompt using the OpenAI Responses session API. This avoids re-sending the heavy instructions every loop and can cut prompt-token costs by ~15–20% in long runs. You can control it via `prompt.cache_system_prompt` in `conf/config.yaml`:

```yaml
prompt:
  cache_system_prompt: true  # set to false to fall back to inline system prompts
```

Caching works automatically for `api_type: "responses"`; if you switch back to the legacy Chat API the flag is ignored.

### Game Start Grace Period

Sequential runs (for example via `run_navigation.ps1`) can ask the API for state faster than the server resets, which previously made the second run exit immediately with `Game completed!`. The new `game_start_wait` knob (default `1.5` seconds) pauses after `start_game` until the API reports a non-completed state:

```yaml
game_start_wait: 1.5  # increase if your server needs longer between levels
```

Set it to `0` if you want the old behavior.

### GPT-5 Reasoning Effort

If you switch the model to any `gpt-5` variant, the agent now sends OpenAI's `reasoning` payload with your preferred effort level. GPT-5 ignores the `temperature` knob, so only the reasoning effort is applied. Control it via the new `thinking_level` field in `conf/config.yaml`:

```yaml
thinking_level: "low"  # options: low, medium, high
```

The default is `low`, matching OpenAI's recommendation for most navigation tasks.

## Local DGX ITB Vision Server (localhost:8001)

Use the `local_dgx_itb` client to talk to our self-hosted server that exposes an OpenAI-compatible `v1/chat/completions` endpoint. This cluster rotates through the following multimodal models, so pick whichever you have loaded:

1. `Qwen/Qwen2-VL-7B-Instruct`
2. `Qwen/Qwen2.5-VL-7B-Instruct`
3. `Qwen/Qwen3-VL-8B-Thinking`
4. `zai-org/GLM-4.1V-9B-Thinking`

```pwsh
$env:VLM_API_KEY = "sk-your-qwen-key"
python src/main.py --config-name local_dgx_itb model="Qwen/Qwen2.5-VL-7B-Instruct"
```

The client sends the same payload structure as this cURL example:

```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
     -H "Authorization: sk-RkcDGuu9Io6bAeDBd7tZFdDKLn67wfH2" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "meta-llama-3-vision",
           "input": [
             {
               "role": "user",
               "content": [
                 {"type": "input_text", "text": "Analyze this situation."},
                 {"type": "input_image", "image_url": "data:image/png;base64,..."}
               ]
             }
           ],
           "max_output_tokens": 256
         }'
```

Sample multi-turn request/response handled by the new client:

**Request**

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_image",
          "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJ...[data_base64_screenshot]...hI7g9/9k="
        },
        {
          "type": "input_text",
          "text": "Berdasarkan kondisi game di layar, apa langkah terbaik selanjutnya? Berikan output dalam format perintah aksi: MOVE_UP, MOVE_DOWN, ATTACK, atau PASS."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "PASS"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Lingkungan berubah, coba nilai kembali."
        }
      ]
    }
  ],
  "max_output_tokens": 128,
  "temperature": 0.8
}
```

**Response**

```json
{
  "object": "response",
  "created_at": 1764042555,
  "status": "completed",
  "error": null,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "This is a simulated response from 'Qwen/Qwen2.5-VL-7B-Instruct'. I processed your text: 'Lingkungan berubah, coba nilai kembali.' and 1 image(s). My recommended action is ATTACK."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 180,
    "output_tokens": 40,
    "total_tokens": 220
  }
}
```

If you need different credentials or hosts, override `llm_base_url`, `llm_api_key`, or `llm_api_key_env` via Hydra CLI flags (for example `llm_base_url=http://dgx02:9000`).

## 🗺️ Start Game with Custom Map Text

Need to send a raw `.txt` map to the API without Hydra configs? Use `script/start_game_with_map.py` (defaults to `localhost:8000` and `data/map_dataset/1_navigation/straight_corridor.txt`):

```pwsh
python script/start_game_with_map.py data/map_dataset/custom_map.txt --host localhost --port 8000
```

The helper reads the file, strips whitespace, and posts it to `/start_game` with `mode: "string"`. Override `map`/`--host`/`--port` as needed.







curl -X POST "http://localhost:8000/start-game" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "procedural",
    "max_rooms": 30,
    "room_min_size": 4,
    "room_max_size": 6,
    "map_width": 30,
    "map_height": 30
  }'