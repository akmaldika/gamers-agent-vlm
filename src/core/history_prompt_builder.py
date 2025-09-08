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

from games.dungeon_escape.prompts import get_fewshot_examples, get_dungeon_system_prompt
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
    
    def __init__(self, max_history_pairs: int = 2, use_fewshot: bool = False, use_visual_tiles: bool = False):
        """
        Initialize PromptManager.
        
        Args:
            max_history_pairs: Maximum number of user-assistant pairs to keep in history
            use_fewshot: Whether to include few-shot examples (False for zero-shot)
            use_visual_tiles: Whether to include visual tile examples in system prompt
        """
        self.max_history_pairs = max_history_pairs
        self.use_fewshot = use_fewshot
        self.use_visual_tiles = use_visual_tiles
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
            formatted_state = f"""Game State:
Dungeon Level: {game_state.get('dungeon_level', '?')}
Steps on current level: {game_state.get('current_level_step_count', '?')}
Player Health: {game_state.get('player_health', '?')}
Health Potions: {game_state.get('health_potion_count', '?')}
Player standing on: {game_state.get('player_standing_on', '?')}

Recent Messages:
{chr(10).join(game_state.get('message_log', ['No messages']))}"""
            
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
        
        # Read game state log
        game_state_log = self.read_game_state_log()
        
        # Build game state text with appropriate header (agent loop vs game step)
        game_state_text = f"CURRENT SITUATION - Loop {step_number}:\n" + game_state_log + "\n"
        
        # NOT USE THIS for now 
        # if additional_context:
        #     game_state_text += f"\nAdditional context: {additional_context}\n"
        
        # Create observation
        observation = Observation(
            step_number=step_number,
            game_state_text=game_state_text,
            image_path=image_path,
            timestamp=datetime.now()
        )
        
        self.observations.append(observation)
        
        # Keep only recent observations to manage memory
        max_observations = self.max_history_pairs + 1  # +1 for current
        if len(self.observations) > max_observations:
            self.observations = self.observations[-max_observations:]
    
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
        )
        
        self.action_responses.append(action_response)
        
        # Keep only recent responses to manage memory
        if len(self.action_responses) > self.max_history_pairs:
            self.action_responses = self.action_responses[-self.max_history_pairs:]
    
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

        # 1. System prompt (text only for API compatibility). If visual tiles are enabled,
        #    move the images into a separate user message.
        if isinstance(self.system_prompt, list):
            # Extract all text parts for the system message
            text_parts = [item.get("text", "") for item in self.system_prompt if item.get("type") == "text"]
            system_text = "\n".join([t for t in text_parts if t])
            messages.append({"role": "system", "content": system_text})

            # Create a separate user message with visual tiles (images allowed on user role)
            tiles_content: List[Dict[str, Any]] = [{
                "type": "text",
                "text": "Reference: Tile types (images + descriptions)"
            }]
            # Skip the first big base prompt text (index 0) to avoid duplication
            for idx, item in enumerate(self.system_prompt):
                if idx == 0:
                    continue
                tiles_content.append(item)
            messages.append({"role": "user", "content": tiles_content})
        else:
            # Plain text system prompt
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

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
                            "image_url": {"url": image_url}
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
                if observation.image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_base64_url(observation.image_data)
                        }
                    })

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
                        "url": self._image_to_base64_url(current_obs.image_data)
                    }
                })

            messages.append({
                "role": "user",
                "content": current_content
            })

            # 5. Add thought process guidance WITH IMAGE
            thought_process_prompt = (
                "Now analyze this current state in phases:\n\n"
                "Phase 1. **What do you see?**\n"
                "   - Identify your character's position, what's on the top, bottom, left, and right\n"
                "   - Count any visible enemies or items.\n\n"
                "Phase 2. **What is the player standing on?**\n"
                "Phase 3. **What should you do based on that?**\n\n"
                "Then respond as:\n"
                "<think>your detailed analysis</think>\n"
                "<action>[a short key sequence like d, d, s]</action>\n"
            )

            thinking_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": thought_process_prompt
                }
            ]

            # Add current image again for analysis
            if current_obs.image_data:
                thinking_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_base64_url(current_obs.image_data)
                    }
                })

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
