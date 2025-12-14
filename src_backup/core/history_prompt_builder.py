"""
History-based Prompt Manager for Dungeon Game AI
Handles conversation history and prompt building for multimodal interactions.
"""

import os
import sys
import json
import base64
import io
from typing import List, Dict, Any, Optional, Union
from PIL import Image as PILImage
from dataclasses import dataclass
from datetime import datetime

# Using proper package imports; no sys.path manipulation needed

# Import game API client for getting game state
try:
    from game_api_client import get_api_client
    _api_available = True
except ImportError:
    # Fallback if import fails
    _api_available = False
    get_api_client = None

from games.dungeon_escape.prompts import get_fewshot_examples, get_dungeon_system_prompt, get_tile_types
# For backward compatibility
try:
    from games.dungeon_escape.prompts import DUNGEON_SYSTEM_PROMPT
except ImportError:
    # Fallback - create a basic system prompt
    DUNGEON_SYSTEM_PROMPT = "You are a dungeon exploration AI agent."


@dataclass
class Observation:
    """Represents a game observation with text and image."""
    step_number: int
    game_state_text: str
    image_path: str
    timestamp: datetime
    image_data: Optional[PILImage.Image] = None
    position: Optional[str] = None  # Player coordinate at this observation
    
    def __post_init__(self):
        """Load image data if not provided."""
        if self.image_data is None and os.path.exists(self.image_path):
            try:
                self.image_data = PILImage.open(self.image_path)
            except Exception as e:
                print(f"Warning: Could not load image {self.image_path}: {e}")

""" 
@dataclass
class ActionResponse:
    # Represents an AI action response with reasoning
    action: str
    reasoning: str
    step_number: int
    timestamp: datetime
    raw_response: str = ""
"""

@dataclass
class ActionResponse:
    """Represents an AI action response with reasoning."""
    actions: List[str]  # Changed from single action to list
    reasoning: str
    step_number: int
    timestamp: datetime
    raw_response: str = ""
    observation_position: Optional[str] = None
    
    # Backward compatibility property
    @property
    def action(self) -> str:
        """Return first action for backward compatibility"""
        return self.actions[0] if self.actions else ""

class PromptManager:
    """
    Manages conversation history and builds prompts for the dungeon game AI.
    Handles multimodal inputs (text + images) and maintains conversation context.
    """
    
    def __init__(self, max_history_pairs: int = 2, use_fewshot: bool = False, use_visual_tiles: bool = False, action_history_count: int = 0):
        """
        Initialize PromptManager.
        
        Args:
            max_history_pairs: Maximum number of user-assistant pairs to keep in history
            use_fewshot: Whether to include few-shot examples (False for zero-shot)
            use_visual_tiles: Whether to include visual tile examples as separate user message
            action_history_count: Number of recent actions to show in prompt (0 to disable)
        """
        self.max_history_pairs = max_history_pairs
        self.use_fewshot = use_fewshot
        self.use_visual_tiles = use_visual_tiles
        self.action_history_count = action_history_count
        self.observations: List[Observation] = []
        self.action_responses: List[ActionResponse] = []
        self.current_step = 0
        
        # Select system prompt based on visual tiles setting
        self.system_prompt = get_dungeon_system_prompt(use_visual_tiles)
            
        # Load few-shot examples with images (only if use_fewshot is True)
        if self.use_fewshot:
            self.fewshot_examples = get_fewshot_examples(mode=1)  # Use mode=1 for image-based examples
            print(f"Loaded {len(self.fewshot_examples)} few-shot examples")
        else:
            self.fewshot_examples = []
            print("Using zero-shot mode (no examples)")
        
        print(f"History pairs: {max_history_pairs}")
        print(f"Mode: {'Few-shot' if use_fewshot else 'Zero-shot'}")
        print(f"Visual tiles: {'Enabled' if use_visual_tiles else 'Disabled'}")
        print(f"Action history: {action_history_count} actions")
    
    def read_game_state_log(self) -> str:
        """Read current game state via API as formatted text."""
        if not _api_available or get_api_client is None:
            return "API not available - using fallback"
        
        try:
            client = get_api_client()
            game_state = client.get_game_state()
            
            if not game_state:
                return "Game state not available"
            

            # Format game state as readable text (aligned with API fields)
            formatted_state = f"""Game Status (this state taken from game api):
Dungeon Level: {game_state.get('dungeon_level', '?')}
Steps on current level: {game_state.get('current_level_step_count', '?')}
Player Health: {game_state.get('player_health', '?')}
Health Potions: {game_state.get('health_potion_count', '?')}
Player standing on: {game_state.get('player_standing_on', '?')}
Player coordinate position: {game_state.get('player_position', '?')}
Ladder position: {game_state.get('stairs', '?')}
Action Keys that only you can do now: {game_state.get('legal_actions', []) if game_state.get('legal_actions') else ''}

Last Messages (from last action):
{chr(10).join(game_state.get('message_log', ['No messages']))}"""
            
            # Format coordinates as (x, y)
            def _fmt(val):
                if isinstance(val, list) and len(val) == 2:
                    return f"({val[0]}, {val[1]})"
                return str(val)

            formatted_state = f"""Game Status (this state taken from game api):
Player standing on: {game_state.get('player_standing_on', '?')}
Player coordinate position [horizontal, vertical]: {_fmt(game_state.get('player_position', '?'))}
Ladder position: {_fmt(game_state.get('stairs', '?'))}
Action Keys that only you can do now: {game_state.get('legal_actions', []) if game_state.get('legal_actions') else ''}

Last Messages (from game, from last action, 1 state before):
{chr(10).join(game_state.get('message_log', ['No messages']))}"""

            # Do not mutate observations here; observation objects are created in update_observation

            return formatted_state
            
        except Exception as e:
            return f"Failed to read game state via API: {e}"
    
    def update_observation(self, step_number: int, image_path: str, additional_context: str = "") -> None:
        """
        Update with a new observation (game state + image).
        
        Args:
            step_number: Current step number
            image_path: Path to the current screenshot
            additional_context: Additional text context
        """
        self.current_step = step_number
        
        # Try to get raw game_state first so we can include concrete player_position and legal_actions
        player_pos = None
        game_state_log = None
        if _api_available and get_api_client is not None:
            try:
                client = get_api_client()
                gs = client.get_game_state()
                if gs:
                    # Helper to format coordinates as (x, y)
                    def _fmt(val):
                        if isinstance(val, list) and len(val) == 2:
                            return f"({val[0]}, {val[1]})"
                        return str(val)

                    # Build a concise formatted state from the raw game_state so logs and prompts match exactly
                    game_state_log = (
                        f"Game State (this state taken from game api):\n"
                        f"Player standing on: {gs.get('player_standing_on', '?')}\n"
                        f"Player coordinate position [horizontal, vertical]: {_fmt(gs.get('player_position', '?'))}\n"
                        f"Ladder coordinate position: {_fmt(gs.get('stairs', '?'))}\n"
                        f"actions that only you can do now: {gs.get('legal_actions', []) if gs.get('legal_actions') else ''}\n\n"
                        f"Messages (from game, from last action, 1 state before):\n{chr(10).join(gs.get('message_log', ['No messages']))}"
                    )
                    if 'player_position' in gs:
                        player_pos = str(gs.get('player_position'))
            except Exception:
                # If API call fails, we'll fallback to the existing reader below
                player_pos = None
                game_state_log = None

        # If we couldn't build game_state_log from raw API, fallback to the reader method
        if game_state_log is None:
            game_state_log = self.read_game_state_log()


        # Build game state text with appropriate header (agent loop vs game step)
        game_state_text = f"CURRENT SITUATION - State {step_number}:\n" + game_state_log + "\n"

        # Create observation
        observation = Observation(
            step_number=step_number,
            game_state_text=game_state_text,
            image_path=image_path,
            timestamp=datetime.now(),
            position=player_pos
        )
        
        self.observations.append(observation)

        # Also print a short diagnostic when positions are missing for this observation
        # if player_pos is None:
        #     print(f"DEBUG: update_observation step={step_number} - player_pos=NONE")
        # else:
        #     print(f"DEBUG: update_observation step={step_number} - player_pos={player_pos}")
        
        # Keep only recent observations to manage memory
        # Ensure we keep enough observations to satisfy both the history pairs
        # requirement and the separate action_history_count (which may be larger).
        # We need at least (max_history_pairs + 1) observations to pair previous
        # observations with actions, and at least (action_history_count + 1)
        # observations so that _get_action_history_text can map recent actions to
        # observations (including the current one).
        retention_obs = max(self.max_history_pairs + 1, self.action_history_count + 1)
        if len(self.observations) > retention_obs:
            self.observations = self.observations[-retention_obs:]
    
    def update_action(self, actions: Union[str, List[str]], reasoning: str, raw_response: str = "") -> None:
        """
        Update with AI's action response.
        
        Args:
            action: The action key (e.g., 'w', 's', 'space')
            reasoning: AI's reasoning for the action
            raw_response: Full raw response from AI
        """
        if isinstance(actions, str):
            actions = [actions]  # Convert single action to list
            
        action_response = ActionResponse(
            actions=actions,
            reasoning=reasoning,
            step_number=self.current_step,
            timestamp=datetime.now(),
            raw_response=raw_response
            ,observation_position=(self.observations[-1].position if self.observations else None)
        )
        
        self.action_responses.append(action_response)
        
        # Keep only recent responses to manage memory. Ensure we retain enough
        # responses to display action history (action_history_count) as well as
        # any configured history pairs.
        retention_actions = max(self.max_history_pairs, self.action_history_count)
        if retention_actions <= 0:
            # If retention_actions is 0, we still want to keep at least the most
            # recent response so that current behavior remains predictable.
            retention_actions = 1

        if len(self.action_responses) > retention_actions:
            self.action_responses = self.action_responses[-retention_actions:]
    
    def reset_action_history(self) -> None:
        """
        Reset action history (called when ascending stairs/changing levels).
        This keeps conversation history but clears action sequence for new level.
        """
        print("Action history reset (new level)")
        # Only reset action responses, keep observations for context
        self.action_responses = []
    
    def _get_action_history_text(self) -> str:
        """
        Generate action history text showing recent actions.
        
        Returns:
            Formatted string of recent actions or empty string if disabled.
        """
        if self.action_history_count <= 0 or not self.action_responses:
            return ""
        
        # Get recent actions (limited by action_history_count)
        recent_responses = self.action_responses[-self.action_history_count:]
        all_actions = []
        positions: List[str] = []

        for response in recent_responses:
            # collect actions
            if hasattr(response, 'actions') and response.actions:
                all_actions.extend(response.actions)
            elif hasattr(response, 'action') and response.action:
                all_actions.append(response.action)

            # try to find observation matching this action's step_number
            pos_val = '?'
            try:
                step_num = int(getattr(response, 'step_number', -1))
            except Exception:
                step_num = -1

            if step_num >= 0:
                # First prefer any position captured and stored directly on the action response
                resp_pos = getattr(response, 'observation_position', None)
                if resp_pos is not None:
                    pos_val = str(resp_pos)
                else:
                    # exact match
                    for obs in self.observations:
                        if getattr(obs, 'step_number', None) == step_num and getattr(obs, 'position', None) is not None:
                            pos_val = str(obs.position)
                            break

                # fallback: nearest earlier observation with position
                if pos_val == '?':
                    candidates = [o for o in self.observations if getattr(o, 'step_number', -9999) <= step_num and getattr(o, 'position', None) is not None]
                    if candidates:
                        best = max(candidates, key=lambda o: getattr(o, 'step_number', -9999))
                        pos_val = str(best.position)

            positions.append(pos_val)

        # Debug: if any positions are unknown, print diagnostics to stdout
        if any(p == '?' for p in positions):
            try:
                print("DEBUG: _get_action_history_text - some positions unknown")
                for idx, resp in enumerate(recent_responses):
                    sn = getattr(resp, 'step_number', None)
                    print(f"  action[{idx}] step={sn} actions={getattr(resp,'actions', None)} -> pos={positions[idx]}")
                print("  Observations stored:")
                for o in self.observations:
                    print(f"    obs step={getattr(o,'step_number', None)} position={getattr(o,'position', None)}")
            except Exception:
                pass
            # Also persist a snapshot to disk for offline diagnosis (won't affect prompt content)
            try:
                dbg_dir = os.path.join("data", "log")
                os.makedirs(dbg_dir, exist_ok=True)
                snap_path = os.path.join(dbg_dir, f"action_history_snapshot_step_{self.current_step}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt")
                with open(snap_path, "w", encoding="utf-8") as sf:
                    sf.write(f"Snapshot Time: {datetime.now().isoformat()}\n")
                    sf.write(f"PromptManager current_step: {self.current_step}\n")
                    sf.write(f"Recent responses (last {len(recent_responses)}):\n")
                    for idx, resp in enumerate(recent_responses):
                        sf.write(f"  [{idx}] step={getattr(resp,'step_number',None)} actions={getattr(resp,'actions',None)} -> pos={positions[idx]}\n")
                    sf.write("Observations stored:\n")
                    for o in self.observations:
                        sf.write(f"  obs step={getattr(o,'step_number',None)} position={getattr(o,'position',None)} game_state_text=\"{str(o.game_state_text)[:200].replace('\n','\\n')}...\"\n")
            except Exception:
                pass

        if not all_actions:
            return ""

        def _format_pos(pos_val: Any) -> str:
            if not isinstance(pos_val, str):
                return str(pos_val)
            cleaned = pos_val.strip().lstrip("[").rstrip("]")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) == 2:
                return f"({parts[0]},{parts[1]})"
            return pos_val

        formatted_positions = [_format_pos(p) for p in positions]

        actions_str = "[" + ",".join(all_actions) + "]"
        positions_str = "[" + ",".join(formatted_positions) + "]"

        return (
            f"Actions that you have done (ascending chronological order): {actions_str}\n"
            # f"Positions that you have traced (chronological order, 1 tile = 8 pixel): {[tuple(p.split(', ')) if isinstance(p, str) else tuple(p) for p in positions]}\n\n"
            f"Positions that you have traced (chronological order, 1 tile = 8 pixel): {positions_str}\n\n"

        )

    def _image_to_base64_url(self, image: PILImage.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        import io
        import base64
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def _image_file_to_base64_url(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 data URL."""
        if not os.path.exists(image_path):
            return None
        
        try:
            with PILImage.open(image_path) as img:
                return self._image_to_base64_url(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_prompt(self) -> List[Dict[str, Any]]:
        """
        Build the complete prompt messages for the AI model using the integrated style.
        Returns a single coherent sequence of messages (text with optional images).
        """
        return self.build_prompt()
    
    def build_prompt(self) -> List[Dict[str, Any]]:
        """
        Build the complete prompt messages for the AI model (integrated style).
        
        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = []

        # 1. System prompt (always text-only now)
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # 2. Add visual tile types as separate user message if enabled
        if self.use_visual_tiles:
            visual_tiles = get_tile_types(1)  # Get visual tile types (mode=1 for images)
            if isinstance(visual_tiles, list):
                tiles_content: List[Dict[str, Any]] = [{
                    "type": "text",
                    "text": "Reference: Tile types (images + descriptions)"
                }]
                tiles_content.extend(visual_tiles)
                messages.append({"role": "user", "content": tiles_content})

        # 2. Optional few-shot examples (single multimodal user message)
        if self.use_fewshot and self.fewshot_examples:
            fewshot_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": "Here are some examples of how to act. Follow this pattern."
                }
            ]

            for i, example in enumerate(self.fewshot_examples):
                fewshot_content.append({
                    "type": "text",
                    "text": f"\n--- Example {i + 1} ---"
                })

                if example.get("image_path"):
                    image_url = self._image_file_to_base64_url(example["image_path"])
                    if image_url:
                        fewshot_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "original_path": example["image_path"]  # Preserve original path for logging
                            }
                        })

                fewshot_content.append({
                    "type": "text",
                    "text": example.get("fewshot", "")
                })

            fewshot_content.append({
                "type": "text",
                "text": "\n--- Now, solve this new situation ---"
            })

            messages.append({
                "role": "user",
                "content": fewshot_content
            })

        # 3. Build conversation history (user-assistant pairs)
        # Pair observations with actions from previous steps
        history_pairs = max(0, min(len(self.observations) - 1, len(self.action_responses), self.max_history_pairs))

        for i in range(history_pairs):
            # Get observation and corresponding action (offset by -1 since current observation doesn't have action yet)
            obs_idx = -(history_pairs - i + 1)  # Get from older to newer
            action_idx = -(history_pairs - i)   # Corresponding action

            if abs(obs_idx) <= len(self.observations) and abs(action_idx) <= len(self.action_responses):
                observation = self.observations[obs_idx]
                action_response = self.action_responses[action_idx]

                # User message (observation)
                user_content: List[Dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": observation.game_state_text
                    }
                ]

                # Add image if available
                # if observation.image_data:
                #     user_content.append({
                #         "type": "image_url",
                #         "image_url": {
                #             "url": self._image_to_base64_url(observation.image_data)
                #         }
                #     })

                messages.append({
                    "role": "user",
                    "content": user_content
                })

                # Assistant message (action response) — always render actions as a list to prime multi-key outputs
                actions_list = action_response.actions if getattr(action_response, 'actions', None) else []
                actions_text = f"[{', '.join(actions_list)}]" if actions_list else "[]"
                assistant_content = f"<think>\n{action_response.reasoning}\n</think>\n<action>{actions_text}</action>"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

        # 4. Current observation (always the last user message)
        if self.observations:
            current_obs = self.observations[-1]

            current_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": current_obs.game_state_text
                }
            ]

            # Add image if available
            if current_obs.image_data:
                current_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_base64_url(current_obs.image_data),
                        "original_path": current_obs.image_path if hasattr(current_obs, 'image_path') else None
                    }
                })

            messages.append({
                "role": "user",
                "content": current_content
            })

            # 5. Add thought process guidance WITH IMAGE
            action_history_text = self._get_action_history_text()
            thought_process_prompt = (
                f"{action_history_text}"
                "With format output, Now analyze this current state in:\n\n"
                "1. **What your action you can do? then Identify the tiles type and coordinate directly adjacent to the player (exactly UP, DOWN, LEFT, and RIGHT — no diagonals).**\n"
                "2. **What is the player standing on?**\n"
                "3. **What is player position now and what player 1-step position before? (give only coordinate)**\n"
                "4. **where is the coordinate ladder (goal)?**\n"
                "\n"
                "Response Format:\n"
                "<think>your analysis(What should you do based on that?)</think>\n"
                "<action>[key]</action>\n"

                # "Then respond:\n"
                # "<think>your analysis (What should you do based on that?)</think>\n"
                # "<action>[a key]</action>\n"
            )

            thinking_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": thought_process_prompt
                }
            ]

            # Do NOT add the image again here; the current observation image is
            # already included in the previous user message. Avoid duplicating
            # the same image in the prompt.

            messages.append({
                "role": "user",
                "content": thinking_content
            })

        return messages
    
    # Removed: structured thinking variant. Integrated style is the only mode.
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state."""
        return {
            "total_observations": len(self.observations),
            "total_actions": len(self.action_responses),
            "current_step": self.current_step,
            "max_history_pairs": self.max_history_pairs,
            "recent_actions": [ar.action for ar in self.action_responses[-5:]] if self.action_responses else []
        }
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.observations.clear()
        self.action_responses.clear()
        self.current_step = 0
