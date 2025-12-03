# Key mapping for common variations that the AI might return
from typing import List, Dict, Any
import base64
import os

dark_floor = "src/games/dungeon_escape/assets/dark_floor.png"
dark_wall = "src/games/dungeon_escape/assets/dark_wall.png"
floor = "src/games/dungeon_escape/assets/floor.png"
ghost = "src/games/dungeon_escape/assets/ghost.png"
ladder = "src/games/dungeon_escape/assets/ladder.png"
player = "src/games/dungeon_escape/assets/player.png"
wall = "src/games/dungeon_escape/assets/wall.png"

# Few-shot examples with images
FEWSHOT_EXAMPLES = [
    {
        "image_path": "src/games/dungeon_escape/assets/fewshot_step_on_the_ladder.png", 
        "caption": "The ladder is just below the player, the player is NOT on the ladder yet",
        "fewshot":"""<think>
The player is just above the ladder. I will move down one step and then press space to descend.
</think>
<action>[s,space]</action>
        """,
    },
    {
        "image_path": "src/games/dungeon_escape/assets/fewshot_explore_chance.png",
        "caption": "up and down player is wall, left and right is floor, there is path to right, right, down and left, down, down, down",
        "fewshot":"""<think>
The player is in a corridor with walls up and down. There is a path to the right and ledt. i will explor to the right first by right, right, down.
</think>
<action>[d,d,s]</action>
        """,
    },
    {
        "image_path": "src/games/dungeon_escape/assets/fewshot_surround_enemy.png",
        "caption": "The player is surrounded by three enemies from right, up, and down. There is a floor tile to the left with only 1 way",
        "fewshot":"""<think>
The player is surrounded by enemies on three sides. to reduce demage i will go to left and face them one by one. I have 4 power attack and ghost have 15 health, so attack 4 times should be enough to kill it. so it 12 attack to kill all of them.
</think>
<action>[a,d,d,d,d,d,d,d,d,d,d,d,d]</action>
        """,
    },
    {
        "image_path": "src/games/dungeon_escape/assets/fewshot_surround_enemy_with_chance.png",
        "caption": "The player is surrounded by three enemies from right, up, and down. There is a floor tile to the left with only 1 way. There is a ledder on down, down, left, left",
        "fewshot":"""<think>
The player is surrounded by enemies on three sides. But, there is a ladder down, down, left, left. my healt still enough to face the enemies. I will attack the ghost down first (down 4 times), then i will go down, down, left, left to the ladder and descend.
</think>
<action>[s,s,s,s,s,s,a,a,space]</action>
        """,
    },
]

def get_fewshot_examples(mode: int = 0) -> List[Dict[str, Any]]:
    """
    Args:
        mode: 0 for text-only examples, 1 for image-based examples.
    """
    if mode == 0:
        return [
            {
                "image_path": None,
                "caption": example["caption"],
                "fewshot": example["fewshot"]
            }
            for example in FEWSHOT_EXAMPLES
        ]
    elif mode == 1:
        # Return full examples with image paths for base64 conversion
        return FEWSHOT_EXAMPLES
    return []


# TILE_TYPES = """
# - player (you): The player character. A person with brown hair and a blue scarf and red pants
# - floor: A walkable tile. Teal-colored with a soft light dot in the center.
# - floor_dark: An unlit/out-of-sight floor tile. A darker version of the standard floor tile.
# - walls : you CANNOT pass this.  A wall made of light grey bricks.
# - wall_dark: you CANNOT pass this: An unlit/out-of-sight wall. A darker, bluer version of the standard wall.
# - ladder: press 'space' to descend when standing on it: A section of a wooden or copper-colored pixel art ladder.
# - ghost: An easy enemy. A classic, light-grey pixel art ghost with its arms raised. You attack it by moving into it.
# """

TILE_TYPES = """
- player (you): The player character. A person with brown hair and a blue scarf and red pants
- floor: A walkable tile. Teal-colored with a soft light dot in the center.
- walls : you CANNOT pass this.  A wall made of light grey bricks.
- ladder: press 'space' to descend ONLY when standing EXACTLY on it (same coordinate): A section of a wooden or copper-colored pixel art ladder.
"""


crab = "src/games/dungeon_escape/assets/crab.png"
# dark_floor = "src/games/dungeon_escape/assets/dark_floor.png"
# dark_wall = "src/games/dungeon_escape/assets/dark_wall.png"
floor = "src/games/dungeon_escape/assets/floor.png"
ghost = "src/games/dungeon_escape/assets/ghost.png"
ladder = "src/games/dungeon_escape/assets/ladder.png"
player = "src/games/dungeon_escape/assets/player.png"
wall = "src/games/dungeon_escape/assets/wall.png"

def _file_to_data_url(path: str) -> str:
    """Load an image file and return a base64 data URL (PNG)."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # Fallback to original path if reading fails
        return path


def get_tile_types(mode: int) -> str | List[Dict[str, Any]]:
    """Generate tile types description.
    Args:
        mode: 0 for text-only examples, 1 for image-based examples.
    """
    if mode == 0:
        return TILE_TYPES
    elif mode == 1:
        # Return structured content for multimodal messages with images embedded as data URLs
        tile_content: List[Dict[str, Any]] = []

        # Pre-convert file paths to data URLs
        player_url = _file_to_data_url(player)
        floor_url = _file_to_data_url(floor)
        wall_url = _file_to_data_url(wall)
        ladder_url = _file_to_data_url(ladder)
        ghost_url = _file_to_data_url(ghost)
        crab_url = _file_to_data_url(crab)

        # Player tile
        tile_content.extend([
            {"type": "text", "text": "This is the Player tile:"},
            {"type": "image_url", "image_url": {"url": player_url, "original_path": player}},
            {"type": "text", "text": "- player (you): The player character. A person with brown hair and a blue scarf and red pants, you have attack power 4\n"}
        ])

        # Floor tile
        tile_content.extend([
            {"type": "text", "text": "This is a Floor tile:"},
            {"type": "image_url", "image_url": {"url": floor_url, "original_path": floor}},
            {"type": "text", "text": "- floor: A walkable tile. Teal-colored with a soft light dot in the center.\n- floor_dark: An unlit/out-of-sight floor tile (darker version of above)\n"}
        ])

        # Wall tile
        tile_content.extend([
            {"type": "text", "text": "This is a Wall tile:"},
            {"type": "image_url", "image_url": {"url": wall_url, "original_path": wall}},
            {"type": "text", "text": "- walls: you CANNOT pass this. A wall made of light grey bricks.\n- wall_dark: you CANNOT pass this. An unlit/out-of-sight wall (darker version of above)\n"}
        ])

        # Ladder tile
        tile_content.extend([
            {"type": "text", "text": "This is a Ladder tile:"},
            {"type": "image_url", "image_url": {"url": ladder_url, "original_path": ladder}},
            {"type": "text", "text": "- ladder: press 'space' to descend ONLY when standing EXACTLY on it (same coordinate). A section of a wooden or copper-colored pixel art ladder.\n"}
        ])

        # # Ghost enemy
        # tile_content.extend([
        #     {"type": "text", "text": "This is a Ghost enemy:"},
        #     {"type": "image_url", "image_url": {"url": ghost_url}},
        #     {"type": "text", "text": "- ghost: An easy enemy. A classic, light-grey pixel art ghost with its arms raised. ghost has 10 health and 3 attack  power\n"}
        # ])

        # # Crab enemy
        # tile_content.extend([
        #     {"type": "text", "text": "This is a Crab enemy:"},
        #     {"type": "image_url", "image_url": {"url": crab_url}},
        #     {"type": "text", "text": "- crab: a hard enemy. A red pixel art crab with large claws and small black eyes. troll has 15 health and 8 attack power\n"}
        # ])

        return tile_content
    
    return TILE_TYPES

ACTIONS = {
    'w': 'move up',
    's': 'move down',
    'a': 'move left',
    'd': 'move right',
    'space': 'descend / take ladder / stairs (ONLY if standing EXACTLY on ladder coordinate)',
    '.': 'wait / skip a turn (do nothing)',
    'g': 'pick up item (only if standing on a wooden box/chest)',
    'i': 'use health potion (if available)'
}

def get_actions_description():
    """Generate a formatted string of all available actions."""
    return "\n".join([f"{key}: {desc}" for key, desc in ACTIONS.items()])

def get_dungeon_system_prompt(use_visual_tiles: bool = False) -> str:
    """Generate the dungeon system prompt.
    Args:
        use_visual_tiles: Kept for compatibility but system prompt is always text-only now.
                         Tile types will be handled separately in user messages.
    """
    base_prompt = f"""## Objective:
You are a highly capable agent tasked with efficiently exploring and descending dungeon levels.
Your PRIMARY OBJECTIVE: Reach and descend the ladder to the next dungeon level as efficiently as possible.

## Instructions:
1. **Identify Key Elements:**
    - Detect your current position as the player.
    - Locate the ladder tile on the map.
    - You will receive a map image and game status in text form.
    - You will also receive possible actions you can take in the current state. (some action not available meaning that direaction are wall)

2. **Ladder Navigation:**
    - If the ladder is visible: Navigate toward it using the shortest walkable path. When your position / coordinate matches the ladder’s position (EXACTLY SAME COORDINATE), press 'space' to descend.
    - If the ladder is NOT visible: Explore to reveal new areas, prioritizing unexplored or dark tiles and minimizing repeat steps.

3. **Avoid:**
    - Repeating the same action sequences or retracing explored routes. (you will provide the coordinates you have visited and the action history so far)
    - Dead-ends or blocked paths. If blocked, seek alternative paths rather than oscillating in the same area.
---

## Action Keys:
{get_actions_description()}


---

## Response Format:
<think>Your reasoning based on the current game state.</think>
<action>[key]</action>
"""

    """ 
    4. **You have limited vision:**
    - You can only see tiles become darker tiles when out of sight
    - if you found pitch black side by side with tiles or dark tiles, there is part of the dungeon that is unexplored (no wall between it)

4. **Limited Vision:**
    - Tiles become dark when out of sight, so prioritize exploration to maintain map awareness.    

    ## Response Format:
<think>
[Your reasoning based on the current game state.]
</think>
<action>[key1, key2, key3, ...]</action>
    """

    # When not using visual tiles, include TILE_TYPES in the system prompt
    if not use_visual_tiles:
        tile_types_section = f"""
---

## Tile Types Reference:
{TILE_TYPES}"""
        base_prompt += tile_types_section

    return base_prompt

# Keep the old constant for backward compatibility
DUNGEON_SYSTEM_PROMPT = get_dungeon_system_prompt(False)


if __name__ == "__main__":
    print(DUNGEON_SYSTEM_PROMPT)


""" 
Example response:

<think>
I see the player. The ladder are 3 step on left. I will move left 3 times and then press space to descend.
</think>
<action>[d,d,d,space]</action>
"""