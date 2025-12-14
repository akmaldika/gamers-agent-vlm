"""
Manual Inference Exporter for Web-based VLM Testing

Exports game state to ./input/ folder for manual testing via web interfaces like GLM-4V.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional


class ManualInferenceExporter:
    """Export game state for manual inference via web chat interface."""
    
    def __init__(self, export_dir: str = "input"):
        """Initialize exporter.
        
        Args:
            export_dir: Directory to export files (default: "input")
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
    def clear_export_dir(self):
        """Clear all files in export directory."""
        if self.export_dir.exists():
            shutil.rmtree(self.export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_state(
        self,
        game_state_image_path: str,
        history_text: Optional[str] = None,
        visual_tile_paths: Optional[List[str]] = None,
        system_prompt: str = "",
        few_shot_examples: Optional[str] = None,
        current_situation_text: Optional[str] = None,
        analysis_prompt: Optional[str] = None,
        step_number: int = 1,
        max_tokens: int = 500
    ):
        """Export current game state for manual inference.
        
        Args:
            game_state_image_path: Path to game screenshot
            history_text: Optional text describing previous loops/actions
            visual_tile_paths: List of paths to tile images (if using visual tiles)
            system_prompt: System instructions
            few_shot_examples: Few-shot example text (if using few-shot mode)
            current_situation_text: Latest observation text to show CURRENT SITUATION
            step_number: Current step number
            max_tokens: Maximum output tokens (used as guideline in prompt)
        """
        # Clear previous state
        self.clear_export_dir()
        
        # Copy game state image
        game_img_dest = self.export_dir / "game_state.png"
        if os.path.exists(game_state_image_path):
            shutil.copy2(game_state_image_path, game_img_dest)
        
        # Copy visual tile images if provided
        tile_names = []
        if visual_tile_paths:
            for tile_path in visual_tile_paths:
                if os.path.exists(tile_path):
                    tile_name = Path(tile_path).name
                    tile_dest = self.export_dir / tile_name
                    shutil.copy2(tile_path, tile_dest)
                    tile_names.append(tile_name)
        
        # Build clean prompt text
        prompt_parts = []
        
        # Add system instructions
        if system_prompt:
            prompt_parts.append("=== INSTRUCTIONS ===")
            # Clean system prompt from XML tags
            clean_system = system_prompt.replace("<content>", "").replace("</content>", "").strip()
            prompt_parts.append(clean_system)
            prompt_parts.append("")
        
        # Add few-shot examples if provided
        if few_shot_examples:
            prompt_parts.append("=== EXAMPLES ===")
            clean_examples = few_shot_examples.replace("<content>", "").replace("</content>", "").strip()
            prompt_parts.append(clean_examples)
            prompt_parts.append("")
        
        # Add image reference section WITHOUT numbering in text
        # (Web models infer image order from upload sequence)
        prompt_parts.append("=== REFERENCE IMAGES ===")
        if tile_names:
            prompt_parts.append("I have uploaded reference images showing tile types.")
            prompt_parts.append("These help you understand what each tile looks like.")
            prompt_parts.append("")
        
        prompt_parts.append("=== CURRENT GAME STATE IMAGE ===")
        prompt_parts.append("The last uploaded image shows the current game state.")
        prompt_parts.append("")
        
        # Build image order list for file references
        image_order = tile_names + ["game_state.png"]
        
        if history_text:
            prompt_parts.append("=== PREVIOUS STEPS ===")
            clean_history = history_text.replace("<content>", "").replace("</content>", "").strip()
            prompt_parts.append(clean_history)
            prompt_parts.append("")

        if current_situation_text:
            prompt_parts.append("=== CURRENT SITUATION ===")
            clean_current = current_situation_text.replace("<content>", "").replace("</content>", "").strip()
            prompt_parts.append(clean_current)
            prompt_parts.append("")
        else:
            prompt_parts.append("=== CURRENT SITUATION ===")
            prompt_parts.append(f"Loop {step_number}")
            prompt_parts.append("")

        if analysis_prompt:
            prompt_parts.append("=== TASK ===")
            clean_task = analysis_prompt.replace("<content>", "").replace("</content>", "").strip()
            prompt_parts.append(clean_task)
            prompt_parts.append("")
        
        # Add response format
        # prompt_parts.append("=== RESPONSE FORMAT ===")
        
        # Add token guideline based on max_tokens
        if max_tokens:
            # Rough estimate: 1 token ≈ 0.75 words
            estimated_words = int(max_tokens * 0.75)
            prompt_parts.append(f"Keep your response concise (approximately {estimated_words} words or less).")
            prompt_parts.append("")
        
        # prompt_parts.append("<think>")
        # prompt_parts.append("[Your reasoning based on the current game state.]")
        # prompt_parts.append("</think>")
        # prompt_parts.append("<action>[key]</action>")
        # prompt_parts.append("")
        
        # Save prompt to file
        prompt_text = "\n".join(prompt_parts)
        prompt_file = self.export_dir / "prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        # Save image order reference
        order_file = self.export_dir / "image_order.txt"
        with open(order_file, 'w', encoding='utf-8') as f:
            f.write("Upload images in this order:\n")
            for i, img_name in enumerate(image_order, 1):
                f.write(f"{i}. {img_name}\n")
        
        # Create README for instructions
        readme_file = self.export_dir / "README.txt"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("=== HOW TO USE ===\n\n")
            f.write("1. Open GLM-4V web interface\n")
            f.write("2. Upload images in order (see image_order.txt):\n")
            for i, img_name in enumerate(image_order, 1):
                f.write(f"   {i}. {img_name}\n")
            f.write("\n3. Copy-paste prompt from prompt.txt\n")
            f.write("\n4. Submit and wait for response\n")
            f.write("\n5. Parse response for <action> tag\n")
            f.write("\n=== FILES ===\n")
            f.write(f"- prompt.txt: Full prompt text\n")
            f.write(f"- image_order.txt: Image upload sequence\n")
            f.write(f"- game_state.png: Current game screenshot\n")
            if tile_names:
                f.write(f"- {', '.join(tile_names)}: Reference tile images\n")
        
        print(f"✓ Exported state to: {self.export_dir.absolute()}")
        print(f"  - {len(image_order)} images")
        print(f"  - prompt.txt ({len(prompt_text)} chars)")
        print(f"  - See README.txt for instructions")


def format_prompt_for_web(
    system_prompt: str,
    tile_descriptions: str,
    game_state_text: str,
    use_visual_tiles: bool = False
) -> str:
    """Format prompt text for web interface (single message box).
    
    Args:
        system_prompt: System instructions
        tile_descriptions: Tile type descriptions
        game_state_text: Current game state
        use_visual_tiles: Whether visual tiles are being used
        
    Returns:
        Clean formatted prompt text
    """
    parts = []
    
    # System instructions
    parts.append("=== YOUR ROLE ===")
    clean_system = system_prompt.replace("<content>", "").replace("</content>", "").strip()
    parts.append(clean_system)
    parts.append("")
    
    # Tile reference
    if not use_visual_tiles:
        parts.append("=== TILE TYPES ===")
        clean_tiles = tile_descriptions.replace("<content>", "").replace("</content>", "").strip()
        parts.append(clean_tiles)
        parts.append("")
    else:
        parts.append("=== REFERENCE IMAGES ===")
        parts.append("See uploaded images for tile type references")
        parts.append("")
    
    # Game state
    parts.append("=== CURRENT GAME STATE ===")
    clean_state = game_state_text.replace("<content>", "").replace("</content>", "").strip()
    parts.append(clean_state)
    parts.append("")
    
    # Response format
    parts.append("=== RESPOND IN THIS FORMAT ===")
    parts.append("<think>")
    parts.append("[Your analysis and reasoning]")
    parts.append("</think>")
    parts.append("<action>[key]</action>")
    
    return "\n".join(parts)
