# Key mapping for common variations that the AI might return
from typing import List, Dict, Any

chest = "src/games/dungeon_escape/assets/chest.png"
crab = "src/games/dungeon_escape/assets/crab.png"
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
<action>[s, space]</action>
        """,
    },
    {
        "image_path": "src/games/dungeon_escape/assets/fewshot_explore_chance.png",
        "caption": "up and down player is wall, left and right is floor, there is path to right, right, down and left, down, down, down",
        "fewshot":"""<think>
The player is in a corridor with walls up and down. There is a path to the right and ledt. i will explor to the right first by right, right, down.
</think>
<action>[d, d, s]</action>
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
<action>[s,s,s,s, s, s, a, a, space]</action>
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


TILE_TYPES = f"""
- player (you): The player character. A person with brown hair and a blue scarf and red pants
- floor: A walkable tile. Teal-colored with a soft light dot in the center.
- floor_dark: An unlit/out-of-sight floor tile. A darker version of the standard floor tile.
- walls : you CANNOT pass this.  A wall made of light grey bricks.
- wall_dark: you CANNOT pass this: An unlit/out-of-sight wall. A darker, bluer version of the standard wall.
- ladder: press 'space' to descend when standing on it: A section of a wooden or copper-colored pixel art ladder.
- chest: A pixel art treasure chest, seemingly made of wood with metallic blue/silver reinforcements.
- ghost: An easy enemy. A classic, light-grey pixel art ghost with its arms raised.
- crab: a hard enemy. A red pixel art crab with large claws and small black eyes.
"""

chest = "src/games/dungeon_escape/assets/chest.png"
crab = "src/games/dungeon_escape/assets/crab.png"
dark_floor = "src/games/dungeon_escape/assets/dark_floor.png"
dark_wall = "src/games/dungeon_escape/assets/dark_wall.png"
floor = "src/games/dungeon_escape/assets/floor.png"
ghost = "src/games/dungeon_escape/assets/ghost.png"
ladder = "src/games/dungeon_escape/assets/ladder.png"
player = "src/games/dungeon_escape/assets/player.png"
wall = "src/games/dungeon_escape/assets/wall.png"

def get_tile_types(mode: int) -> str | List[Dict[str, Any]]:
    """Generate tile types description.
    Args:
        mode: 0 for text-only examples, 1 for image-based examples.
    """
    if mode == 0:
        return "\n".join([f"- {desc}" for desc in TILE_TYPES])
    elif mode == 1:
        pass
    return "\n".join([f"- {desc}" for desc in TILE_TYPES])

ACTIONS = {
    'w': 'move up',
    's': 'move down',
    'a': 'move left',
    'd': 'move right',
    'space': 'descend / take ladder (stairs)',
    '.': 'wait / skip a turn',
    'g': 'pick up item (only if standing on a chest)',

}

def get_actions_description():
    """Generate a formatted string of all available actions."""
    return "/n".join([f"{key}: {desc}" for key, desc in ACTIONS.items()])

DUNGEON_SYSTEM_PROMPT = f"""
## Objective:
You are an intelligent speedrun agent playing a dungeon exploration game.
Your PRIMARY OBJECTIVE is to reach and descend the ladder to the next dungeon level as efficiently as possible.

## Additional Instructions:
1. **Identify Key Elements:**
   - Detect your current position (represented by a character on the map).
   - Scan the map for the ladder tile (brown with black).

2. **If ladder Are Visible:**
   - Navigate directly to the ladder using the shortest walkable path.
   - When standing on the ladder, press 'space' to descend.

3. **If ladder Are NOT Visible:**
   - Explore systematically to reveal unexplored/dark areas.
   - Avoid retracing your steps unnecessarily.
   - Seek out room entrances and corridors that lead to new areas.

4. **You have limited vision:**
    - You can only see tiles become dark tiles when out of sight
    - if you found pitch black side by side with tiles or dark tiles, there is part of the dungeon that is unexplored (no wall between it)

---

## Tile Types:
{TILE_TYPES}

## Action Keys:
{get_actions_description()}

---

## Response Format:
<think>
[Your reasoning based on the current game state.]
</think>
<action>[key1, key2, key3, ...]</action>

Example response:

<think>
I see the player. The ladder are 3 step on left. I will move left 3 times and then press space to descend.
</think>
<action>[d,d,d,space]</action>
"""


if __name__ == "__main__":
    print(DUNGEON_SYSTEM_PROMPT)
    