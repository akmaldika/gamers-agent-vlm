import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleGameLogger:
    """Simple JSON-based game logger."""
    
    def __init__(self, game_name: str = "dungeon_escape_ai", base_dir: str = "data/log", base_path: Optional[str] = None):
        """
        Initialize the game logger.
        
        Args:
            game_name: Name of the game (for compatibility)
            base_dir: Base directory for logs (legacy)
            base_path: New base path for logs (takes priority over base_dir)
        """
        self.game_name = game_name
        self.base_dir = Path(base_path if base_path else base_dir)
        self.game_number = self._get_next_game_number()
        self.session_dir = self.base_dir / f"{self.game_number}-game"
        self.images_dir = self.session_dir / "images"
        
        # Create directories
        self._setup_directories()
        
        # Session metadata
        self.session_start = datetime.now()
        self.session_data = {
            "game_name": game_name,
            "game_number": self.game_number,
            "session_id": self.session_dir.name,
            "session_start": self.session_start.isoformat(),
            "session_end": None,
            "duration_seconds": None,
            "strategy": None,
            "system_prompt": None,
            "total_steps": 0,
            "steps": []
        }
        
        print(f"📂 Session created: {self.session_dir}")
    
    def _get_next_game_number(self) -> int:
        """Get the next game number by checking existing directories."""
        if not self.base_dir.exists():
            return 1
        
        existing_games = [
            int(d.name.split('-')[0]) 
            for d in self.base_dir.iterdir() 
            if d.is_dir() and d.name.endswith('-game') and d.name.split('-')[0].isdigit()
        ]
        
        return max(existing_games, default=0) + 1
    
    def _setup_directories(self):
        """Create all necessary directories."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
    
    def set_strategy(self, strategy: str):
        """Set the game strategy."""
        self.session_data["strategy"] = strategy
        print(f"📝 Strategy set: {strategy}")
    
    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt (only once)."""
        if self.session_data["system_prompt"] is None:
            self.session_data["system_prompt"] = system_prompt
            print(f"📝 System prompt set (length: {len(system_prompt)} chars)")
    
    def log_step(self, step_number: int, input_messages: List[Dict], output_response: Any, 
                 parsed_action: Dict, image_path: Optional[str] = None):
        """
        Log a step with pure input-output format.
        
        Args:
            step_number: The step number
            input_messages: List of message dicts sent to OpenAI [{"role": "user", "content": "..."}]
            output_response: OpenAI response object (with model_dump_json capability)
            parsed_action: {"reasoning": "...", "action": "w"}
            image_path: Path to the screenshot image
        """
        
        # Copy image to logs directory if provided
        logged_image_path = None
        if image_path and os.path.exists(image_path):
            logged_image_path = self.images_dir / f"step_{step_number}.png"
            shutil.copy2(image_path, logged_image_path)
            print(f"🖼️  Image saved: {logged_image_path}")
        
        # Convert OpenAI response to dict
        if hasattr(output_response, 'model_dump'):
            # Pydantic v2 style
            output_dict = output_response.model_dump()
        elif hasattr(output_response, 'dict'):
            # Pydantic v1 style
            output_dict = output_response.dict()
        elif hasattr(output_response, '_asdict'):
            # namedtuple style (our LLMResponse)
            output_dict = output_response._asdict()
        else:
            # Fallback: try to convert to dict
            output_dict = {
                "model_id": getattr(output_response, 'model_id', 'unknown'),
                "completion": getattr(output_response, 'completion', ''),
                "stop_reason": getattr(output_response, 'stop_reason', 'unknown'),
                "input_tokens": getattr(output_response, 'input_tokens', 0),
                "output_tokens": getattr(output_response, 'output_tokens', 0)
            }
        
        # Store input messages in original format (same as sent to LLM)
        # This preserves multimodal structure for debugging
        
        # Store system prompt once (extract from first message if it's system)
        if input_messages and input_messages[0]["role"] == "system":
            if self.session_data["system_prompt"] is None:
                self.session_data["system_prompt"] = input_messages[0]["content"]
        
        # Create step data with original format
        step_data = {
            "step_number": step_number,
            "timestamp": datetime.now().isoformat(),
            "input_messages": input_messages,  # Keep original format with multimodal content
            "output_response": output_dict,
            "parsed_action": parsed_action,
            "image_path": str(logged_image_path.relative_to(self.session_dir)) if logged_image_path else None
        }
        
        self.session_data["steps"].append(step_data)
        self.session_data["total_steps"] = step_number
        
        # Save updated session data
        self._save_session_data()
        
        # Save individual input/output files
        self._save_input_output_files(step_number, input_messages, output_response, image_path)
        
        print(f"📝 Step {step_number} logged (JSON format + input/output files)")
    
    def _save_input_output_files(self, step_number: int, input_messages: List[Dict], 
                                output_response: Any, image_path: Optional[str] = None):
        """
        Save individual input and output files for a step.
        
        Args:
            step_number: The step number
            input_messages: Raw input messages sent to model
            output_response: Raw output response from model
            image_path: Path to image if any
        """
        # Save output file (raw completion)
        output_file = self.session_dir / f"output_{step_number}.txt"
        output_content = getattr(output_response, 'completion', str(output_response))
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        # Save input file (structured format with tags)
        input_file = self.session_dir / f"input_{step_number}.txt"
        input_content = self._format_input_messages(input_messages, image_path)
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(input_content)
        
        print(f"📄 Saved input_{step_number}.txt and output_{step_number}.txt")
    
    def _format_input_messages(self, input_messages: List[Dict], image_path: Optional[str] = None) -> str:
        """
        Format input messages with XML-like tags for better structure.
        
        Args:
            input_messages: List of message dicts
            image_path: Path to image if any
            
        Returns:
            Formatted string with tags
        """
        formatted_content = []
        
        for msg in input_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, list):
                # Multimodal content
                formatted_content.append(f"<{role}_prompt>")
                
                for item in content:
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        formatted_content.append("<content>")
                        formatted_content.append(text_content)
                        formatted_content.append("</content>")
                    elif item.get("type") == "image_url":
                        formatted_content.append("<content>")
                        if image_path:
                            formatted_content.append(f"{{image_path: {image_path}}}")
                        else:
                            formatted_content.append("{image_attachment}")
                        formatted_content.append("</content>")
                
                formatted_content.append(f"</{role}_prompt>")
            else:
                # Simple text content
                formatted_content.append(f"<{role}_prompt>")
                formatted_content.append("<content>")
                formatted_content.append(str(content))
                formatted_content.append("</content>")
                formatted_content.append(f"</{role}_prompt>")
        
        return "\n".join(formatted_content)
    
    def _save_session_data(self):
        """Save session data to JSON file."""
        session_file = self.session_dir / "game_session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
    
    def finalize_session(self, summary_data: Optional[Dict[str, Any]] = None):
        """
        Finalize the logging session.
        
        Args:
            summary_data: Optional additional summary information
        """
        self.session_data["session_end"] = datetime.now().isoformat()
        self.session_data["duration_seconds"] = (
            datetime.now() - self.session_start
        ).total_seconds()
        
        # Add summary data if provided
        if summary_data:
            self.session_data.update(summary_data)
        
        # Save final session data
        self._save_session_data()
        
        print(f"✅ JSON session finalized: {self.session_dir}")
        print(f"📊 Total steps logged: {self.session_data['total_steps']}")
        print(f"⏱️  Duration: {self.session_data['duration_seconds']:.1f} seconds")
    
    def get_session_path(self) -> str:
        """Get the current session directory path."""
        return str(self.session_dir)
    
    def get_conversation_format(self) -> List[Dict]:
        """
        Get conversation in pure message format for analysis.
        
        Returns:
            List of message dictionaries in chronological order
        """
        conversation = []
        
        # Add system prompt if exists
        if self.session_data["system_prompt"]:
            conversation.append({
                "role": "system",
                "content": self.session_data["system_prompt"]
            })
        
        # Add all steps as user-assistant pairs
        for step in self.session_data["steps"]:
            # Add user messages (skip system since already added)
            for msg in step["input_messages"]:
                if msg["role"] != "system":
                    conversation.append(msg)
            
            # Add assistant response
            conversation.append({
                "role": "assistant", 
                "content": step["output_response"].get("completion", "")
            })
        
        return conversation
