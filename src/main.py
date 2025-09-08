import os
import time
import re
from typing import List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from collections import namedtuple

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from core.client import OpenAIWrapper
from games.dungeon_escape.tools import (
    get_game_screenshot,
    send_key,
    get_game_state,
    start_game_from_file,
)
from games.dungeon_escape.prompts import ACTIONS
from core.history_prompt_builder import PromptManager
from core.simple_game_logger import SimpleGameLogger
from game_api_client import get_api_client

load_dotenv()

@dataclass
class GameStep:
    step_number: int
    action: str
    timestamp: datetime
    reasoning: str = ""
    actions: Optional[List[str]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = [self.action]


TestMessage = namedtuple("TestMessage", ["role", "content", "attachment"])


class DungeonAgent:
    def __init__(self, api_key: str, logger: Optional[SimpleGameLogger] = None, config: Optional[DictConfig] = None):
        from types import SimpleNamespace

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
                "temperature": temperature,
            },
            max_retries=3,
            delay=1,
        )

        self.client = OpenAIWrapper(client_config)
        self.config = config
        self.steps: List[GameStep] = []
        self.current_step = 0  # Agent loop counter (resets per level)
        self.current_level: Optional[int] = None
        self.logger = logger

        try:
            history_count = config.prompt.history_count if config and hasattr(config, "prompt") else 0
            use_fewshot = config.prompt.use_fewshot if config and hasattr(config, "prompt") else False
            use_visual_tiles = config.prompt.use_visual_tiles if config and hasattr(config, "prompt") else False
        except AttributeError:
            history_count = 0
            use_fewshot = False
            use_visual_tiles = False

        history_count = max(0, min(5, history_count))

        self.prompt_manager = PromptManager(
            max_history_pairs=history_count,
            use_fewshot=use_fewshot,
            use_visual_tiles=use_visual_tiles,
        )

    def check_game_completion(self) -> bool:
        try:
            api_game_state = get_game_state()
            if not api_game_state:
                return False
            return bool(api_game_state.get("is_done", False))
        except Exception as e:
            print(f"Error checking game completion: {e}")
            return False

    def get_action_with_react(self, image_path: str, game_state: str, step_number: int = 0) -> tuple[List[str], List[dict], Any, dict]:
        self.prompt_manager.update_observation(
            step_number=step_number,
            image_path=image_path,
            additional_context=game_state,
        )

        messages = self.prompt_manager.get_prompt()

        if self.logger and messages and messages[0]["role"] == "system":
            self.logger.set_system_prompt(messages[0]["content"])

        test_messages = []
        input_messages_for_log = messages.copy()

        for msg in messages:
            if isinstance(msg["content"], list):
                text_content = ""
                image_content = None
                for content_item in msg["content"]:
                    if content_item["type"] == "text":
                        text_content += content_item["text"]
                    elif content_item["type"] == "image_url":
                        if step_number > 0:
                            from PIL import Image as PILImage
                            try:
                                image_content = PILImage.open(image_path)
                            except Exception as e:
                                print(f"Warning: Could not load image {image_path}: {e}")
                test_messages.append(TestMessage(role=msg["role"], content=text_content, attachment=image_content))
            else:
                test_messages.append(TestMessage(role=msg["role"], content=msg["content"], attachment=None))

        response = self.client.generate(test_messages)
        raw_response = response.completion.strip() if response.completion else ""
        reasoning, actions = self.parse_ai_response(raw_response)

        if actions and reasoning:
            self.prompt_manager.update_action(actions, reasoning, raw_response)

        parsed_action = {"reasoning": reasoning, "actions": actions}
        return actions, input_messages_for_log, response, parsed_action

    def add_step(self, action: str, reasoning: str = ""):
        step = GameStep(step_number=self.current_step + 1, action=action, timestamp=datetime.now(), reasoning=reasoning)
        self.steps.append(step)
        self.current_step += 1
        return step

    def print_step_summary(self):
        print("\nGame Session Summary")
        print(f"Total Loops: {self.current_step}")
        game_completed = self.check_game_completion()
        print("Game Status: COMPLETED SUCCESSFULLY!" if game_completed else "Game Status: Session ended (not completed)")
        print("\nAll Loops:")
        for step in self.steps:
            print(f"Loop {step.step_number}: '{step.action}' at {step.timestamp.strftime('%H:%M:%S')}")
        if self.logger:
            action_counts: dict[str, int] = {}
            for step in self.steps:
                action_counts[step.action] = action_counts.get(step.action, 0) + 1
            summary_data = {
                "final_loop_count": self.current_step,
                "action_frequencies": action_counts,
                "game_status": "completed" if game_completed else "session_ended",
                "game_completed": game_completed,
                "steps_summary": [
                    {
                        "loop_number": step.step_number,
                        "action": step.action,
                        "timestamp": step.timestamp.isoformat(),
                        "reasoning": step.reasoning,
                    }
                    for step in self.steps
                ],
            }
            self.logger.finalize_session(summary_data)

    def autonomous_game_loop(self, window_title: str = "Dungeon Escape AI", max_steps: int = 100, map_file_path: str | None = None, custom_map: str | None = None):
        print("Dungeon Game Agent - Autonomous Mode")
        print(f"Agent will play automatically for max {max_steps} loops per level")

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
        self.current_level = initial_state.get("dungeon_level")
        print(f"Initial state: Level {self.current_level}, Health {initial_state.get('player_health')}")
        print("Press Ctrl+C to stop manually\n")

        try:
            while True:
                try:
                    current_game_state = get_game_state()
                except Exception as e:
                    print(f"Error getting game state: {e}")
                    current_game_state = None

                if current_game_state and bool(current_game_state.get("is_done", False)):
                    print("Game completed! Completion detected in game state.")
                    print("Agent stopping automatically...")
                    break

                if current_game_state:
                    level = current_game_state.get("dungeon_level")
                    if self.current_level is None:
                        self.current_level = level
                    elif level is not None and level != self.current_level:
                        print(f"Level changed: {self.current_level} -> {level}. Resetting loop counter.")
                        self.current_level = level
                        self.current_step = 0

                if self.current_step >= max_steps:
                    print(f"Reached max loops ({max_steps}) for level {self.current_level}. Stopping.")
                    break

                step_num = self.current_step + 1
                print(f"--- Loop {step_num} (Level {self.current_level}) ---")

                # Save screenshot directly into the logger's images directory
                if self.logger and hasattr(self.logger, 'images_dir'):
                    images_dir = Path(self.logger.images_dir)
                else:
                    images_dir = Path("current")
                images_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = images_dir / f"loop_{step_num}.png"
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

                if current_game_state:
                    game_context = (
                        f"Dungeon Level: {current_game_state.get('dungeon_level', '?')}, "
                        f"Game step: {current_game_state.get('current_level_step_count', '?')}, "
                        f"Health: {current_game_state.get('player_health', '?')}, "
                        f"Potions: {current_game_state.get('health_potion_count', '?')}, "
                        f"Standing on: {current_game_state.get('player_standing_on', '?')}"
                    )
                    messages = current_game_state.get("message_log", [])
                    if messages:
                        game_context += f"\nRecent messages: {', '.join(messages[-3:])}"
                    game_state = (
                        f"Current situation: {game_context}. Find the fastest path to ladder (stairs). "
                        f"If ladder not visible, explore systematically."
                    )
                else:
                    game_state = "Analyze the dungeon carefully. Find the fastest path to ladder (stairs)."

                try:
                    actions, input_messages, output_response, parsed_action = self.get_action_with_react(
                        str(screenshot_path), game_state, step_num
                    )
                    reasoning = parsed_action.get("reasoning", "No reasoning provided")
                    print(f"AI decided actions: {actions} (Total: {len(actions)})")
                    print(f"AI Reasoning: {reasoning}")
                    if any(action.lower() in ["esc", "quit", "game over"] for action in actions):
                        print("Game over detected by AI")
                        break
                    if not actions:
                        print("No valid actions received, skipping step...")
                        continue
                except Exception as e:
                    print(f"Error analyzing image: {e}")
                    continue

                try:
                    all_successful = True
                    for i, action in enumerate(actions):
                        print(f"Executing action {i+1}/{len(actions)}: {action}")
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
                        if i < len(actions) - 1:
                            time.sleep(0.2)

                    if self.logger and all_successful:
                        self.logger.log_step(step_num, input_messages, output_response, parsed_action, str(screenshot_path))

                    for action in actions:
                        step = self.add_step(action, reasoning)
                        print(f"Loop {step.step_number} completed: Action '{action}' processed")
                except Exception as e:
                    error_msg = f"Error executing actions: {e}"
                    print(error_msg)
                    if self.logger:
                        error_parsed_action = {
                            "reasoning": parsed_action.get("reasoning", ""),
                            "actions": parsed_action.get("actions", []),
                            "error": str(e),
                        }
                        self.logger.log_step(step_num, input_messages, output_response, error_parsed_action, str(screenshot_path))
                    continue

                time.sleep(1)
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
        reasoning = ""
        actions: List[str] = []
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        action_match = re.search(r"<action>\[(.*?)\]</action>", raw_response, re.DOTALL)
        if action_match:
            actions_str = action_match.group(1).strip()
            parts = [p.strip() for p in actions_str.split(',')]
            actions = [p.strip("'\"") for p in parts if p]
        else:
            action_match = re.search(r"<action>(.*?)</action>", raw_response, re.DOTALL)
            if action_match:
                single_action = action_match.group(1).strip()
                actions = [single_action] if single_action else []
        return reasoning, actions


def get_next_session_id(log_path: str) -> int:
    """Get the next session ID by checking existing log directories."""
    if not os.path.exists(log_path):
        return 1

    existing_dirs = [d for d in os.listdir(log_path) if d.endswith('-game')]
    if not existing_dirs:
        return 1

    session_numbers = []
    for dir_name in existing_dirs:
        try:
            session_num = int(dir_name.split('-')[0])
            session_numbers.append(session_num)
        except (ValueError, IndexError):
            continue

    return max(session_numbers, default=0) + 1


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("Dungeon Escape AI - Runner")
    print(f"Config loaded:\n{OmegaConf.to_yaml(cfg)}")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set; relying on default client auth")

    print(f"API URL: {cfg.api_url}")
    api_client = get_api_client(cfg.api_url)
    if not api_client.check_connection():
        print("ERROR: Cannot connect to game API server. Please ensure the server is running.")
        return
    else:
        print("CONNECTED: Game API server is ready")

    map_file_path = None
    if cfg.map_file:
        map_file_path = cfg.map_file
        if not os.path.exists(map_file_path):
            print(f"ERROR: Map file '{map_file_path}' not found.")
            return
        print(f"Loading map from: {map_file_path}")
    else:
        print("Using procedural generation")

    print(f"Max steps per level: {cfg.max_steps}")

    session_id = get_next_session_id(cfg.log.path)
    log_name = cfg.log.naming.replace("{session_id}", str(session_id))
    logger = SimpleGameLogger(game_name=log_name, base_path=cfg.log.path, use_custom_name=True)
    print(f"Logging to: {logger.get_session_path()}")

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


if __name__ == "__main__":
    main()