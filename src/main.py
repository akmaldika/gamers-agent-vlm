import os
import time
import re
from typing import List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import namedtuple

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Project imports
from dotenv import load_dotenv
from core.client import OpenAIWrapper
from games.dungeon_escape.tools import get_game_screenshot, send_key, get_game_state, start_game_from_file
from games.dungeon_escape.prompts import ACTIONS
from core.history_prompt_builder import PromptManager
from core.simple_game_logger import SimpleGameLogger
from game_api_client import get_api_client

load_dotenv()

@dataclass
class GameStep:
    """Represents a single step in the game."""
    step_number: int
    action: str  # Keep as single action for backward compatibility
    timestamp: datetime
    reasoning: str = ""
    actions: Optional[List[str]] = None  # Add optional field for multiple actions
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = [self.action]

# Create a simple message object for the client
TestMessage = namedtuple("TestMessage", ["role", "content", "attachment"])

class DungeonAgent:
    def __init__(self, api_key: str, logger: Optional[SimpleGameLogger] = None, config: Optional[DictConfig] = None):
        # Initialize OpenAI client using the wrapper
        from types import SimpleNamespace
        
        # Use config if provided, otherwise use defaults
        if config:
            model_id = config.model
            temperature = config.temperature
            max_tokens = config.max_tokens
        else:
            model_id = "gpt-4o-mini"
            temperature = 0.1
            max_tokens = 500
        
        client_config = SimpleNamespace(
            client_name="openai",
            model_id=model_id,
            base_url=None,
            timeout=30,
            generate_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            max_retries=3,
            delay=1
        )
        
        self.client = OpenAIWrapper(client_config)
        self.config = config
        self.steps: List[GameStep] = []
        self.current_step = 0
        self.logger = logger
        
        # Initialize PromptManager with fixed settings
        self.prompt_manager = PromptManager(max_history_pairs=0, strategy="speedrun", thinking_style="integrated")
        
        if self.logger:
            self.logger.set_strategy("speedrun")
    
    def check_game_completion(self) -> bool:
        """Check if the game is completed by examining game state via API."""
        try:
            # Get game state from API
            api_game_state = get_game_state()
            if not api_game_state:
                return False
                
            # Check message log for completion indicators
            message_log = api_game_state.get('message_log', [])
            message_text = ' '.join(message_log).lower()
            
            completion_indicators = [
                "game done",
                "congratulation",
                "press 'q' to quit",
                "congratulation",
                "victory",
                "completed",
                "game over",
                "you died",
            ]
            
            for indicator in completion_indicators:
                if indicator in message_text:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking game completion: {e}")
            return False

    def get_action_with_react(self, image_path: str, game_state: str, step_number: int = 0) -> tuple[List[str], List[dict], Any, dict]:
        """Get action using ReAct approach with PromptManager handling multimodal context."""
        # Update observation in PromptManager
        self.prompt_manager.update_observation(
            step_number=step_number,
            image_path=image_path,
            additional_context=game_state
        )
        
        # Get the complete prompt messages
        messages = self.prompt_manager.get_prompt()
        
        # Set system prompt in logger (only once)
        if self.logger and messages and messages[0]["role"] == "system":
            self.logger.set_system_prompt(messages[0]["content"])
        
        # Convert messages to TestMessage format for client
        test_messages = []
        # Keep original messages format for logging (same as sent to LLM)
        input_messages_for_log = messages.copy()  # Use original format
        
        for msg in messages:
            if isinstance(msg["content"], list):
                # Handle multimodal content (text + image)
                text_content = ""
                image_content = None
                
                for content_item in msg["content"]:
                    if content_item["type"] == "text":
                        text_content += content_item["text"]
                    elif content_item["type"] == "image_url":
                        # For TestMessage format, we need PIL Image
                        # The image is already loaded in PromptManager
                        # We'll use the current observation's image
                        if step_number > 0:
                            from PIL import Image as PILImage
                            try:
                                image_content = PILImage.open(image_path)
                            except Exception as e:
                                print(f"Warning: Could not load image {image_path}: {e}")
                
                test_messages.append(TestMessage(
                    role=msg["role"],
                    content=text_content,
                    attachment=image_content
                ))
                
            else:
                # Simple text content
                test_messages.append(TestMessage(
                    role=msg["role"],
                    content=msg["content"],
                    attachment=None
                ))
        
        # Use the client wrapper
        response = self.client.generate(test_messages)
        
        raw_response = response.completion.strip() if response.completion else ""
        
        # Parse the response to extract reasoning and actions
        reasoning, actions = self.parse_ai_response(raw_response)  # Now returns List[str]
        
        # Update action in PromptManager
        if actions and reasoning:
            self.prompt_manager.update_action(actions, reasoning, raw_response)  # Already supports List[str]
        
        # Prepare parsed action for logging
        parsed_action = {
            "reasoning": reasoning,
            "actions": actions  # Changed from "action" to "actions"
        }
        
        return actions, input_messages_for_log, response, parsed_action  # Return List[str]

    def add_step(self, action: str, reasoning: str = ""):
        """Add a step to the history."""
        step = GameStep(
            step_number=self.current_step + 1,
            action=action,
            timestamp=datetime.now(),
            reasoning=reasoning,
        )
        self.steps.append(step)
        self.current_step += 1
        return step

    def print_step_summary(self):
        """Print summary of all steps taken and finalize logging."""
        print(f"\nGame Session Summary")
        print(f"Total Steps: {self.current_step}")
        
        # Check if game was completed
        game_completed = self.check_game_completion()
        if game_completed:
            print(f"Game Status: COMPLETED SUCCESSFULLY!")
        else:
            print(f"Game Status: Session ended (not completed)")
        
        print(f"\nAll Steps Taken:")
        for step in self.steps:
            print(f"Step {step.step_number}: '{step.action}' at {step.timestamp.strftime('%H:%M:%S')}")
        
        # Finalize logging if logger is available
        if self.logger:
            # Create summary data
            action_counts = {}
            for step in self.steps:
                action = step.action
                action_counts[action] = action_counts.get(action, 0) + 1
            
            summary_data = {
                "final_step_count": self.current_step,
                "action_frequencies": action_counts,
                "game_status": "completed" if game_completed else "session_ended",
                "game_completed": game_completed,
                "steps_summary": [
                    {
                        "step_number": step.step_number,
                        "action": step.action,
                        "timestamp": step.timestamp.isoformat(),
                        "reasoning": step.reasoning
                    }
                    for step in self.steps
                ]
            }
            
            self.logger.finalize_session(summary_data)

    def autonomous_game_loop(self, window_title: str = "Dungeon Escape AI", max_steps: int = 100, map_file_path: str | None = None, custom_map: str | None = None):
        """Autonomous game loop - agent plays by itself using API."""
        print(f"Dungeon Game Agent - Autonomous Mode")
        print(f"Agent will play automatically for max {max_steps} steps")
        
        # Initialize game via API
        if map_file_path:
            print(f"Starting game with map: {map_file_path}")
            initial_state = start_game_from_file(map_file_path)
        elif custom_map:
            print("Starting game with custom map string")
            api_client = get_api_client()
            initial_state = api_client.start_game("string", custom_map=custom_map)
        else:
            print("Starting game with procedural generation")
            api_client = get_api_client()
            initial_state = api_client.start_game("procedural")
            
        if not initial_state:
            print("FAILED: Could not start game via API")
            return False
            
        print("SUCCESS: Game started successfully!")
        print(f"Initial state: Level {initial_state.get('dungeon_level')}, Health {initial_state.get('player_health')}")
        print("Press Ctrl+C to stop manually\n")
        
        try:
            while self.current_step < max_steps:
                step_num = self.current_step + 1
                print(f"--- Step {step_num} ---")
                
                # 1. Take screenshot automatically
                screenshot_path = f"current/step_{step_num}.png"
                try:
                    image = get_game_screenshot()
                    if image:
                        image.save(screenshot_path)
                        print(f"Screenshot captured: {screenshot_path}")
                    else:
                        print("Failed to capture screenshot")
                        break
                except Exception as e:
                    print(f"Error capturing screenshot: {e}")
                    break
                
                # 1.5. Check if game is completed
                if self.check_game_completion():
                    print("Game completed! Completion detected in game state.")
                    print("Agent stopping automatically...")
                    break
                
                # 2. Get current game state for context
                try:
                    current_game_state = get_game_state()
                    if current_game_state:
                        game_context = (
                            f"Level: {current_game_state.get('dungeon_level', '?')}, "
                            f"Step: {current_game_state.get('current_level_step_count', '?')}, "
                            f"Health: {current_game_state.get('player_health', '?')}, "
                            f"Potions: {current_game_state.get('health_potion_count', '?')}, "
                            f"Standing on: {current_game_state.get('player_standing_on', '?')}"
                        )
                        
                        # Include recent messages
                        messages = current_game_state.get('message_log', [])
                        if messages:
                            game_context += f"\nRecent messages: {', '.join(messages[-3:])}"
                    else:
                        game_context = "Game state unavailable"
                        
                    game_state = f"Current situation: {game_context}. Find the fastest path to ladder (stairs). If ladder not visible, explore systematically."
                    
                except Exception as e:
                    print(f"Error getting game state: {e}")
                    game_state = "Analyze the dungeon carefully. Find the fastest path to ladder (stairs)."
                
                # 3. Analyze the image and decide action
                try:
                    # Get action with logging data
                    actions, input_messages, output_response, parsed_action = self.get_action_with_react(screenshot_path, game_state, step_num)

                    # Extract reasoning from parsed action
                    reasoning = parsed_action.get('reasoning', 'No reasoning provided')

                    print(f"AI decided actions: {actions}")
                    print(f"AI Reasoning: {reasoning}")
                    print(f"Chosen Actions: {actions} (Total: {len(actions)})")

                    # Check for game over conditions
                    if any(action.lower() in ['esc', 'quit', 'game over'] for action in actions):
                        print("Game over detected by AI")
                        break

                    if not actions:  # Check if actions list is empty
                        print("No valid actions received, skipping step...")
                        continue
                    
                except Exception as e:
                    print(f"Error analyzing image: {e}")
                    continue
                
                # 3. Execute each action in sequence
                try:
                    all_successful = True
                    for i, action in enumerate(actions):
                        print(f"Executing action {i+1}/{len(actions)}: {action}")
                        
                        # Use ACTIONS dict for direct key mapping
                        mapped_key = action.lower()
                        execution_successful = False
                        
                        if mapped_key in ACTIONS:
                            print(f"Sending key: {mapped_key} for action: {ACTIONS[mapped_key]}")
                            send_key(mapped_key, window_title)
                            execution_successful = True
                        else:
                            print(f"Unknown action '{mapped_key}', trying direct key press")
                            send_key(mapped_key, window_title)
                            execution_successful = True
                        
                        if not execution_successful:
                            all_successful = False
                        
                        print(f"Action {i+1}/{len(actions)}: {action} -> {'✅' if execution_successful else '❌'}")
                        
                        # Add delay between actions if needed
                        if i < len(actions) - 1:  # Don't delay after last action
                            time.sleep(0.2)  # Short delay between multiple actions
                    
                    # Log all actions as one step if logger is available
                    if self.logger and all_successful:
                        self.logger.log_step(step_num, input_messages, output_response, parsed_action, screenshot_path)
                    
                    # Record all actions as individual steps or as one combined step
                    for action in actions:
                        step = self.add_step(action, f"{game_state}")
                        print(f"Step {step.step_number} completed: Action '{action}' processed")
                    
                except Exception as e:
                    error_msg = f"Error executing actions: {e}"
                    print(error_msg)
                    
                    # Log error if logger is available
                    if self.logger:
                        error_parsed_action = {
                            "reasoning": parsed_action.get("reasoning", ""),
                            "actions": parsed_action.get("actions", []),
                            "error": str(e)
                        }
                        self.logger.log_step(step_num, input_messages, output_response, error_parsed_action, screenshot_path)
                    continue
                
                # 4. Wait a bit before next action
                time.sleep(1)  # 1 second delay between actions
                print()
                
        except KeyboardInterrupt:
            print("\nGame interrupted by user!")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        finally:
            self.print_step_summary()
            
        return True

    def parse_ai_response(self, raw_response: str) -> tuple[str, List[str]]:
        """
        Parse AI response to extract reasoning and actions.
        Now returns a list of actions instead of single action.
        """
        reasoning = ""
        actions = []
        
        # Extract reasoning
        think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        
        # Extract actions
        action_match = re.search(r'<action>\[(.*?)\]</action>', raw_response, re.DOTALL)
        if action_match:
            actions_str = action_match.group(1).strip()
            # Split by comma and clean up
            actions = [action.strip().strip('\'"') for action in actions_str.split(',')]
            actions = [action for action in actions if action]  # Remove empty actions
        else:
            # Fallback: try single action format
            action_match = re.search(r'<action>(.*?)</action>', raw_response, re.DOTALL)
            if action_match:
                single_action = action_match.group(1).strip()
                actions = [single_action] if single_action else []
        
        return reasoning, actions

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration management."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        return
    
    print("*** Dungeon Game Agent - API Mode ***")
    print(f"Model: {cfg.model}")
    print(f"Max tokens: {cfg.max_tokens}, Temperature: {cfg.temperature}")
    print(f"API URL: {cfg.api_url}")
    
    # Check API connection
    api_client = get_api_client(cfg.api_url)
    if not api_client.check_connection():
        print("ERROR: Cannot connect to game API server. Please ensure the server is running.")
        return
    else:
        print("CONNECTED: Game API server is ready")
    
    # Determine game mode and map
    map_file_path = None
    
    if cfg.map_file:
        map_file_path = cfg.map_file
        if not os.path.exists(map_file_path):
            print(f"ERROR: Map file '{map_file_path}' not found.")
            return
        print(f"Loading map from: {map_file_path}")
    else:
        print("Using procedural generation")
    
    print(f"Max steps: {cfg.max_steps}")
    
    # Initialize logger with session numbering
    logger = None
    session_id = get_next_session_id(cfg.log.path)
    log_name = cfg.log.naming.replace("{session_id}", str(session_id))
    
    logger = SimpleGameLogger(game_name=log_name, base_path=cfg.log.path)
    print(f"Logging to: {logger.get_session_path()}")
    
    # Initialize and run agent
    try:
        agent = DungeonAgent(api_key=api_key, logger=logger, config=cfg)
        
        result = agent.autonomous_game_loop(max_steps=cfg.max_steps, map_file_path=map_file_path)
            
        if result:
            print("Game completed successfully!")
        else:
            print("Game ended without completion")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def get_next_session_id(log_path: str) -> int:
    """Get the next session ID by checking existing log directories."""
    if not os.path.exists(log_path):
        return 1
    
    existing_dirs = [d for d in os.listdir(log_path) if d.endswith('-game')]
    if not existing_dirs:
        return 1
    
    # Extract numbers from directory names
    session_numbers = []
    for dir_name in existing_dirs:
        try:
            session_num = int(dir_name.split('-')[0])
            session_numbers.append(session_num)
        except (ValueError, IndexError):
            continue
    
    return max(session_numbers, default=0) + 1


if __name__ == "__main__":
    main()