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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import game API client for getting game state
try:
    from game_api_client import get_api_client
    _api_available = True
except ImportError:
    # Fallback if import fails
    _api_available = False
    get_api_client = None

from games.dungeon_escape.prompts import DUNGEON_SYSTEM_PROMPT, get_fewshot_examples


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
    
    def __init__(self, max_history_pairs: int = 2, strategy: str = "speedrun", thinking_style: str = "integrated"):
        """
        Initialize PromptManager.
        
        Args:
            max_history_pairs: Maximum number of user-assistant pairs to keep in history
            strategy: Game strategy (determines system prompt)
            thinking_style: How to structure thinking prompts. Options:
                - "integrated": Single thought process prompt (default)
                - "structured": Separate prompts for each thinking step
        """
        self.max_history_pairs = max_history_pairs
        self.strategy = strategy
        self.thinking_style = thinking_style
        self.observations: List[Observation] = []
        self.action_responses: List[ActionResponse] = []
        self.current_step = 0
        
        # Select system prompt based on strategy
        if strategy == "speedrun":
            self.system_prompt = DUNGEON_SYSTEM_PROMPT
        else:
            # Default fallback is speedrun
            self.system_prompt = DUNGEON_SYSTEM_PROMPT
            
        # Load few-shot examples with images
        self.fewshot_examples = get_fewshot_examples(mode=1)  # Use mode=1 for image-based examples
        print(f"🖼️ Loaded {len(self.fewshot_examples)} few-shot examples")
        print(f"🧠 Using {thinking_style} thinking style")
        print(f"📝 Strategy set: {strategy}")
    
    def read_game_state_log(self) -> str:
        """Read current game state via API as formatted text."""
        if not _api_available or get_api_client is None:
            return "API not available - using fallback"
        
        try:
            client = get_api_client()
            game_state = client.get_game_state()
            
            if not game_state:
                return "Game state not available"
            
            # Format game state as readable text
            formatted_state = f"""Game State:
Level: {game_state.get('dungeon_level', '?')}
Steps on current level: {game_state.get('current_level_step_count', '?')}
Player Health: {game_state.get('player_health', '?')}
Health Potions: {game_state.get('health_potion_count', '?')}
Standing on: {game_state.get('player_standing_on', '?')}

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
        
        # Build game state text with appropriate header
        game_state_text = f"GAME STATE - Step {step_number}:\n"
        game_state_text += game_state_log + "\n"
        
        if additional_context:
            game_state_text += f"\nAdditional context: {additional_context}"
        
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
        Build the complete prompt messages for the AI model.
        Uses the thinking_style specified in constructor.
        
        Returns:
            List of message dictionaries for the AI model
        """
        if self.thinking_style == "structured":
            return self.get_prompt_with_structured_thinking()
        else:
            return self.get_prompt_with_integrated_thinking()
    
    def get_prompt_with_integrated_thinking(self) -> List[Dict[str, Any]]:
        """
        Build the complete prompt messages for the AI model.
        
        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = []
        
        # 1. System prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 2. Add few-shot examples in single message format
        if self.fewshot_examples:
            fewshot_content = []
            
            # Introduction text
            fewshot_content.append({
                "type": "text",
                "text": "Here are some examples of how to act. Follow this pattern."
            })
            
            # Add each example
            for i, example in enumerate(self.fewshot_examples):
                # Example header
                fewshot_content.append({
                    "type": "text",
                    "text": f"\n--- Example {i + 1} ---"
                })
                
                # Add example image
                if example.get('image_path'):
                    image_url = self._image_file_to_base64_url(example['image_path'])
                    if image_url:
                        fewshot_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                
                # Add example response
                fewshot_content.append({
                    "type": "text",
                    "text": example.get('fewshot', '')
                })
            
            # Final instruction
            fewshot_content.append({
                "type": "text",
                "text": "\n--- Now, solve this new situation ---"
            })
            
            # Add the single few-shot message
            messages.append({
                "role": "user",
                "content": fewshot_content
            })
        
        # 3. Build conversation history (user-assistant pairs)
        # We need to pair observations with actions from previous steps
        history_pairs = min(len(self.observations) - 1, len(self.action_responses), self.max_history_pairs)
        
        for i in range(history_pairs):
            # Get observation and corresponding action (offset by -1 since current observation doesn't have action yet)
            obs_idx = -(history_pairs - i + 1)  # Get from older to newer
            action_idx = -(history_pairs - i)   # Corresponding action
            
            if abs(obs_idx) <= len(self.observations) and abs(action_idx) <= len(self.action_responses):
                observation = self.observations[obs_idx]
                action_response = self.action_responses[action_idx]
                
                # User message (observation)
                user_content = []
                
                # Add text content
                user_content.append({
                    "type": "text",
                    "text": observation.game_state_text
                })
                
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
                
                # Assistant message (action response)
                assistant_content = f"<think>\n{action_response.reasoning}\n</think>\n<action>{action_response.action}</action>"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
        
        # 4. Current observation (always the last user message)
        if self.observations:
            current_obs = self.observations[-1]
            
            # Change label for current observation to be more clear
            current_text = current_obs.game_state_text.replace("GAME STATE", "CURRENT SITUATION")
            
            current_content = []
            
            # Add text content
            current_content.append({
                "type": "text",
                "text": current_text
            })
            
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
            
            # 5. FIXED: Add structured thought process guidance WITH IMAGE
            thought_process_prompt = """Now analyze this current state step by step:

1. **What do you see?** 
   - Describe the dungeon layout, walls, floors, and any objects
   - Identify your character's position
   - Count any visible enemies or items, 

2. **What is the player standing on?**
   - Identify the exact tile type under your character
   - Is it a floor, ladder, chest?

3. **What should you do based on that?**
   - If ladder are visible: plan the shortest path to reach them
   - If ladder are not visible: decide which direction to explore

Please follow the response format: <think>your detailed analysis</think><action>key</action>"""

            # FIXED: Include image in thinking prompt
            thinking_content = []
            
            thinking_content.append({
                "type": "text",
                "text": thought_process_prompt
            })
            
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
    
    def get_prompt_with_structured_thinking(self) -> List[Dict[str, Any]]:
        """
        Build prompt with highly structured thinking process using separate user messages.
        This version breaks down the thinking into individual steps for more guided reasoning.
        """
        # Start with the regular prompt (without the thought process)
        messages = []
        
        # 1. System prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 2. Add few-shot examples with images
        for i, example in enumerate(self.fewshot_examples):
            # User message with example image
            user_content = []
            
            # # Add example description and situation
            # user_content.append({
            #     "type": "text",
            #     "text": f"EXAMPLE SITUATION: {example['description']}\n\n{example['explanation']}\n\nWhat should I do in this situation?"
            # })
            
            # Add example image
            if example.get('image_url'):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": example['image_url']
                    }
                })
            
            messages.append({
                "role": "user", 
                "content": user_content
            })
            
            # Assistant response with example reasoning and action
            assistant_content = f"<think>\n{example['example_reasoning']}\n</think>\n<action>{example['suggested_action']}</action>"
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        # 3. Build conversation history (user-assistant pairs)
        history_pairs = min(len(self.observations) - 1, len(self.action_responses), self.max_history_pairs)
        
        for i in range(history_pairs):
            obs_idx = -(history_pairs - i + 1)
            action_idx = -(history_pairs - i)
            
            if abs(obs_idx) <= len(self.observations) and abs(action_idx) <= len(self.action_responses):
                observation = self.observations[obs_idx]
                action_response = self.action_responses[action_idx]
                
                # User message (observation)
                user_content = []
                user_content.append({
                    "type": "text",
                    "text": observation.game_state_text
                })
                
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
                
                # Assistant message (action response)
                assistant_content = f"<think>\n{action_response.reasoning}\n</think>\n<action>{action_response.action}</action>"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
        
        # 4. Current observation
        if self.observations:
            current_obs = self.observations[-1]
            
            # Change label for current observation to be more clear
            current_text = current_obs.game_state_text.replace("GAME STATE", "CURRENT SITUATION")
            
            current_content = []
            current_content.append({
                "type": "text",
                "text": current_text
            })
            
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
            
            # 5. Structured thinking prompts (separate messages for each step)
            thinking_prompts = [
                "Step 1: Carefully examine the image. What do you see in the dungeon? Describe the layout, walls, floors, your character position, and any objects or enemies visible.",
                
                "Step 2: Identify what tile the player character is currently standing on. Is it regular floor tiles, stairs (ladder), a chest, or something else?",
                
                "Step 3: Based on your observations and the speedrun strategy, what action should you take? If stairs are visible, plan the shortest path to reach them. If not, choose the best exploration direction to find unexplored areas."
            ]
            
            for prompt in thinking_prompts:
                # Add each step prompt with image for better analysis
                step_content = []
                step_content.append({
                    "type": "text",
                    "text": prompt
                })
                
                # Add current image for each step analysis
                if current_obs.image_data:
                    step_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_base64_url(current_obs.image_data)
                        }
                    })
                
                messages.append({
                    "role": "user",
                    "content": step_content
                })
            
            # Final instruction
            messages.append({
                "role": "user",
                "content": "Now provide your complete analysis and action using the format: <think>your detailed reasoning combining all observations</think><action>key</action>"
            })
        
        return messages
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state."""
        return {
            "total_observations": len(self.observations),
            "total_actions": len(self.action_responses),
            "current_step": self.current_step,
            "max_history_pairs": self.max_history_pairs,
            "strategy": self.strategy,
            "recent_actions": [ar.action for ar in self.action_responses[-5:]] if self.action_responses else []
        }
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.observations.clear()
        self.action_responses.clear()
        self.current_step = 0
