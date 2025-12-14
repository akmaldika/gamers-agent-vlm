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

from core.client import OpenAIWrapper, ITBDGXClient
from core.manual_inference_exporter import ManualInferenceExporter
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
            # New Config Structure
            client_cfg = config.client
            agent_cfg = config.agent
            params_cfg = agent_cfg.params
            
            model_id = client_cfg.model
            temperature = params_cfg.temperature
            max_tokens = params_cfg.max_output_tokens
            thinking_level = getattr(params_cfg, "thinking_level", "low")
            
            provider = client_cfg.provider
            base_url = client_cfg.base_url
            api_type = getattr(client_cfg, "api_type", "chat")
            
            rate_limit_delay = getattr(params_cfg, "rate_limit_delay", 2.0)
            
            # API Key Logic
            api_key_env = getattr(client_cfg, "api_key_env", None)
            if provider == "openai":
                api_key_value = os.getenv("OPENAI_API_KEY", "")
            elif provider == "local_dgx_itb":
                # Use configured env var or default to ITB_DGX_API_KEY
                env_var_name = api_key_env if api_key_env else "ITB_DGX_API_KEY"
                api_key_value = os.getenv(env_var_name, "")
            else:
                api_key_value = os.getenv("OPENAI_API_KEY", "")
                
        else:
            # Fallback defaults
            model_id = "gpt-4.1-mini"
            temperature = 0.1
            max_tokens = 500
            thinking_level = "low"
            provider = "openai"
            base_url = None
            api_type = "chat"
            rate_limit_delay = 2.0
            api_key_value = os.getenv("OPENAI_API_KEY", "")

        thinking_level = (thinking_level or "low").lower()
        if thinking_level not in {"low", "medium", "high"}:
            thinking_level = "low"

        client_generate_kwargs = {}

        model_lower = model_id.lower()
        using_gpt5 = model_lower.startswith("gpt-5")

        if using_gpt5:
            if api_type == "chat":
                client_generate_kwargs["max_completion_tokens"] = max_tokens
                client_generate_kwargs["reasoning_effort"] = thinking_level
            else:
                client_generate_kwargs["max_output_tokens"] = max_tokens
                client_generate_kwargs["reasoning"] = {"effort": thinking_level}
        else:
            client_generate_kwargs["temperature"] = temperature
            client_generate_kwargs["max_output_tokens"] = max_tokens

        client_config = SimpleNamespace(
            client_name=provider,
            model_id=model_id,
            base_url=base_url,
            timeout=30,
            generate_kwargs=client_generate_kwargs,
            max_retries=3,
            delay=1,
            api_type=api_type,
            rate_limit_delay=rate_limit_delay,
            api_key=api_key_value,
        )

        if provider == "local_dgx_itb":
            self.client = ITBDGXClient(client_config)
        else:
            self.client = OpenAIWrapper(client_config)
            
        self.config = config
        self.steps: List[GameStep] = []
        self.current_step = 0  # Agent loop counter (resets per level)
        self.current_level: Optional[int] = None
        self.logger = logger
        self.game_start_wait = getattr(config.game.execution, "start_wait", 1.5) if config and hasattr(config, "game") else 1.5

        try:
            history_count = config.prompt.history_count if config and hasattr(config, "prompt") else 0
            use_fewshot = config.prompt.use_fewshot if config and hasattr(config, "prompt") else False
            use_visual_tiles = config.prompt.use_visual_tiles if config and hasattr(config, "prompt") else False
            action_history_count = config.prompt.action_history_count if config and hasattr(config, "prompt") else 0
        except AttributeError:
            history_count = 0
            use_fewshot = False
            use_visual_tiles = False
            action_history_count = 0

        history_count = max(0, min(5, history_count))
        action_history_count = max(0, action_history_count)

        self.prompt_manager = PromptManager(
            max_history_pairs=history_count,
            use_fewshot=use_fewshot,
            use_visual_tiles=use_visual_tiles,
            action_history_count=action_history_count,
        )

        prompt_cfg = getattr(config, "prompt", None) if config else None
        cache_system_prompt = bool(getattr(prompt_cfg, "cache_system_prompt", False)) if prompt_cfg else False
        
        # IMPORTANT: System prompt caching via Responses sessions is DISABLED by default.
        supports_session_cache = hasattr(self.client, "ensure_prompt_session")
        self.use_system_prompt_cache = cache_system_prompt and api_type == "responses" and supports_session_cache
        self.prompt_session_id: Optional[str] = None
        self._session_cache_error_logged = False
        
        if self.use_system_prompt_cache:
            print("⚠️  System prompt caching is ENABLED. Ensure system prompt is stable for this session.")
            self.prompt_session_id = self._ensure_prompt_session(initial_attempt=True)
        else:
            # System prompt will be included inline with each request (no session caching)
            pass
        
        # Manual inference mode (for web-based VLM testing)
        self.manual_mode = config.agent.manual_mode if config and hasattr(config, "agent") else False
        self.manual_exporter = ManualInferenceExporter(export_dir="input") if self.manual_mode else None
        
        if self.manual_mode:
            print("=" * 60)
            print("⚠️  MANUAL INFERENCE MODE ENABLED")
            print("=" * 60)
            print("Game state will be exported to ./input/ folder")
            print("You need to manually test via web interface")
            print("=" * 60)

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
        model_messages, session_id = self._prepare_model_messages(messages)

        if self.logger and messages and messages[0]["role"] == "system":
            self.logger.set_system_prompt(messages[0]["content"])

        # MANUAL INFERENCE MODE: Export state and wait for manual input
        if self.manual_mode and self.manual_exporter:
            return self._handle_manual_inference(messages, image_path, game_state, step_number)

        test_messages = []
        input_messages_for_log = messages.copy()

        for msg in model_messages:
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

        response = self.client.generate(test_messages, session_id=session_id)
        raw_response = response.completion if response.completion else ""
        
        # Log token usage to monitor rate limits
        if hasattr(response, 'input_tokens') and hasattr(response, 'output_tokens'):
            total_tokens = response.input_tokens + response.output_tokens
            print(f"Token usage: {response.input_tokens} in + {response.output_tokens} out = {total_tokens} total")
        
        # print(self._format_model_output_for_display(raw_response))
        reasoning, actions = self.parse_ai_response(raw_response)

        if actions and reasoning:
            self.prompt_manager.update_action(actions, reasoning, raw_response)

        parsed_action = {"reasoning": reasoning, "actions": actions}
        return actions, input_messages_for_log, response, parsed_action
    
    def _prepare_model_messages(self, messages: List[dict]) -> tuple[List[dict], Optional[str]]:
        """Drop the system message when a Responses session already stores it."""
        if not self.use_system_prompt_cache:
            return messages, None

        session_id = self.prompt_session_id or self._ensure_prompt_session()
        if not session_id:
            return messages, None

        if messages and messages[0].get("role") == "system":
            trimmed_messages = messages[1:]
        else:
            trimmed_messages = messages

        if not trimmed_messages:
            trimmed_messages = messages

        return trimmed_messages, session_id

    def _ensure_prompt_session(self, initial_attempt: bool = False) -> Optional[str]:
        """Ensure a Responses session exists that stores the system prompt instructions."""
        if not self.use_system_prompt_cache:
            return None

        instructions = getattr(self.prompt_manager, "system_prompt", None)
        if not instructions:
            return None

        session_id = self.client.ensure_prompt_session(instructions)
        if session_id:
            self.prompt_session_id = session_id
            if initial_attempt:
                print("System prompt caching enabled via Responses session instructions.")
            return session_id

        if not self._session_cache_error_logged:
            print("Warning: Failed to create a Responses session for cached system prompt. Falling back to inline prompts.")
            self._session_cache_error_logged = True

        # Disable caching for the rest of the run to avoid repeated API calls.
        self.use_system_prompt_cache = False
        self.prompt_session_id = None
        return None

    def _wait_for_active_game_state(self, initial_state: Optional[dict]) -> Optional[dict]:
        """Wait briefly until the API reports a non-completed game state."""
        wait_total = max(0.0, getattr(self, "game_start_wait", 0.0))
        if wait_total <= 0:
            return initial_state

        def is_active(state: Optional[dict]) -> bool:
            return bool(state) and not state.get("is_done", False)

        if is_active(initial_state):
            return initial_state

        print(f"Waiting up to {wait_total:.1f}s for the dungeon API to load the next map...")
        poll_interval = min(0.5, max(0.1, wait_total / 5))
        deadline = time.time() + wait_total
        last_state = initial_state

        while time.time() < deadline:
            state = get_game_state()
            if state:
                last_state = state
                if is_active(state):
                    return state
            time.sleep(poll_interval)

        if last_state and last_state.get("is_done", False):
            print("Warning: game still reports completion after grace period. Continuing anyway.")
        return last_state
    
    def _handle_manual_inference(self, messages: List[dict], image_path: str, game_state: str, step_number: int) -> tuple[List[str], List[dict], Any, dict]:
        """Handle manual inference mode: export state and wait for user input."""
        
        # Extract components from messages
        system_prompt = ""
        few_shot_text = ""
        history_text = ""
        few_shot_images = []
        analysis_prompt_text = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # System prompt
            if role == "system":
                system_prompt = content if isinstance(content, str) else ""
            
            # User messages (could be few-shot, history, or current state)
            elif role == "user":
                if isinstance(content, list):
                    # Multimodal content
                    text_parts = []
                    for item in content:
                        if item["type"] == "text" or item.get("type") == "input_text":
                            text_parts.append(item["text"])
                        elif item.get("type") in ["image_url", "input_image"]:
                            # Check if this is few-shot image (has original_path with "fewshot_")
                            image_url_obj = item.get("image_url", {})
                            if isinstance(image_url_obj, dict):
                                original_path = image_url_obj.get("original_path", "")
                                if "fewshot_" in original_path:
                                    few_shot_images.append(original_path)
                    
                    combined_text = "\n".join(text_parts)
                    
                    # Determine if this is few-shot or history
                    if "Example" in combined_text or "examples" in combined_text.lower():
                        few_shot_text += combined_text + "\n\n"
                    elif "Loop" in combined_text or step_number > 1:
                        # This might be history
                        if "Loop" in combined_text and f"Loop {step_number}" not in combined_text:
                            history_text += combined_text + "\n\n"

                    if "Now analyze" in combined_text or "<think>" in combined_text:
                        analysis_prompt_text = combined_text.strip()
                else:
                    # Simple text
                    if "Example" in content or "examples" in content.lower():
                        few_shot_text += content + "\n\n"
                    if "Now analyze" in content or "<think>" in content:
                        analysis_prompt_text = content.strip()
        
        # Collect visual tile image paths
        visual_tile_paths = []
        if self.prompt_manager.use_visual_tiles:
            tile_assets_dir = Path("src/games/dungeon_escape/assets")
            for tile_name in ["player.png", "floor.png", "wall.png", "ladder.png"]:
                tile_path = tile_assets_dir / tile_name
                if tile_path.exists():
                    visual_tile_paths.append(str(tile_path))
        
        # Add few-shot images if any
        all_reference_images = visual_tile_paths + [img for img in few_shot_images if os.path.exists(img)]
        
        history_section = history_text.strip() if history_text else ""
        
        # Export state with all context
        max_tokens = getattr(self.config.agent.params, 'max_output_tokens', 500) if self.config else 500
        observation_text = ""
        if self.prompt_manager.observations:
            observation_text = self.prompt_manager.observations[-1].game_state_text.strip()

        self.manual_exporter.export_state(
            game_state_image_path=image_path,
            history_text=history_section or None,
            visual_tile_paths=all_reference_images,
            system_prompt=system_prompt,
            few_shot_examples=few_shot_text.strip() if few_shot_text else None,
            current_situation_text=(observation_text or game_state).strip(),
            analysis_prompt=analysis_prompt_text if analysis_prompt_text else None,
            step_number=step_number,
            max_tokens=max_tokens
        )
        
        print("\n" + "=" * 60)
        print(f" State exported to ./input/ folder (Step {step_number})")
        print("=" * 60)
        print("\n Instructions:")
        print("1. Check ./input/README.txt for upload instructions")
        print("2. Upload images to GLM-4V in the order specified")
        print("3. Copy-paste prompt from ./input/prompt.txt")
        print("4. Get response from model")
        print("5. Enter the response below\n")
        print("=" * 60)
        
        # Wait for user to input response
        print("\n  Paste the model's response (end with empty line):")
        response_lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            response_lines.append(line)
        
        raw_response = "\n".join(response_lines)
        
        # Parse response
        reasoning, actions = self.parse_ai_response(raw_response)
        
        if actions and reasoning:
            self.prompt_manager.update_action(actions, reasoning, raw_response)
        
        # Create mock response object
        from types import SimpleNamespace
        mock_response = SimpleNamespace(
            completion=raw_response,
            model_id="manual-glm4v",
            stop_reason="manual_input",
            input_tokens=0,
            output_tokens=0,
            reasoning=None
        )
        
        parsed_action = {"reasoning": reasoning, "actions": actions}
        return actions, messages, mock_response, parsed_action

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

    def autonomous_game_loop(self, window_title: str = "Dungeon Escape AI", max_steps: int = 100, map_file_path: str | None = None, custom_map: str | None = None, base_url: str = "http://localhost:8000"):
        print("Dungeon Game Agent - Autonomous Mode")
        print(f"Agent will play automatically for max {max_steps} loops per level")

        non_procedural_run = bool(map_file_path or custom_map)

        if map_file_path:
            print(f"Starting game with map: {map_file_path}")
            initial_state = start_game_from_file(map_file_path)
        elif custom_map:
            print("Starting game with custom map string")
            api_client = get_api_client(base_url)
            initial_state = api_client.start_game("string", custom_map=custom_map)
        else:
            print("Starting game with procedural generation")
            api_client = get_api_client(base_url)
            initial_state = api_client.start_game("procedural")

        if not initial_state:
            print("FAILED: Could not start game via API")
            return False

        initial_state = self._wait_for_active_game_state(initial_state)

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
                        # Reset action history for new level
                        self.prompt_manager.reset_action_history()

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
                        game_context += f"\nRecent messages (1 step before): {', '.join(messages[-3:])}"
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
                        print(f"Action {i+1}/{len(actions)}: {action} -> {'TRUE' if execution_successful else '❌'}")
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

                time.sleep(getattr(self.config.game.execution, 'step_delay', 1.0))  # Configurable delay to avoid rate limits
                print()
        except KeyboardInterrupt:
            print("\nGame interrupted by user!")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        finally:
            self.print_step_summary()
            if non_procedural_run:
                self._send_escape_key(window_title)
        return True

    def _send_escape_key(self, window_title: str) -> None:
        try:
            print("Sending ESC to cleanly exit manual map run...")
            send_key('esc', window_title)
        except Exception as exc:
            print(f"Warning: Failed to send ESC key: {exc}")

    @staticmethod
    def _format_model_output_for_display(raw_response: str) -> str:
        if not raw_response:
            return ""
        newline_ratio = raw_response.count("\n") / max(1, len(raw_response))
        if newline_ratio > 0.3:
            # Remove single newlines that are directly followed by non-whitespace
            return re.sub(r"\n(?=\S)", "", raw_response)
        return raw_response

    def parse_ai_response(self, raw_response: str) -> tuple[str, List[str]]:
        def _normalize_tags(text: str) -> str:
            def _clean(match: re.Match) -> str:
                tag = match.group(0)
                # Collapse whitespace/newlines inside the tag name so `<\nt\nh...>` becomes `<think>`
                return re.sub(r"\s+", "", tag)

            return re.sub(r"<[^>]+>", _clean, text)

        def _normalize_action_token(token: str) -> str:
            if not token:
                return ""
            cleaned = token.replace("\r", "").replace("\n", "").strip()
            # Remove nested brackets like [[w]]
            while cleaned.startswith("[") and cleaned.endswith("]"):
                inner = cleaned[1:-1].strip()
                if not inner:
                    break
                cleaned = inner
            cleaned = cleaned.strip("'\"")
            return cleaned

        normalized_response = _normalize_tags(raw_response.strip())
        reasoning = ""
        actions: List[str] = []
        think_match = re.search(r"<think>(.*?)</think>", normalized_response, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        action_match = re.search(r"<action>\[(.*?)\]</action>", normalized_response, re.DOTALL)
        if action_match:
            actions_str = action_match.group(1).strip()
            parts = [p.strip() for p in actions_str.split(',')]
            actions = [_normalize_action_token(p) for p in parts if p]
        else:
            action_match = re.search(r"<action>(.*?)</action>", normalized_response, re.DOTALL)
            if action_match:
                single_action = _normalize_action_token(action_match.group(1).strip())
                actions = [single_action] if single_action else []
        actions = [a for a in actions if a]
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
    # print(f"Config loaded:\n{OmegaConf.to_yaml(cfg)}")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set; relying on default client auth")

    game_base_url = cfg.game.base_url
    print(f"API URL: {game_base_url}")
    api_client = get_api_client(game_base_url)
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

    max_loops = cfg.agent.max_loops
    print(f"Max steps per level: {max_loops}")

    model_log_path = str(Path(cfg.log.path) / cfg.client.model)
    session_id = get_next_session_id(model_log_path)
    log_name = cfg.log.naming.replace("{session_id}", str(session_id)).replace("{model_name}", cfg.client.model)
    logger = SimpleGameLogger(game_name=log_name, base_path=model_log_path, use_custom_name=True)
    print(f"Logging to: {logger.get_session_path()}")

    try:
        agent = DungeonAgent(api_key=api_key, logger=logger, config=cfg)
        result = agent.autonomous_game_loop(
            max_steps=max_loops, 
            map_file_path=map_file_path,
            base_url=game_base_url
        )
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