
"""
Dungeon Game AI Tools - API Version

This module provides tools for interacting with the dungeon game via API,
replacing direct window interaction with HTTP API calls.
"""

from PIL import Image
import sys
import os

# Add the parent directory to sys.path to import game_api_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from game_api_client import GameAPIClient, get_api_client


def get_game_screenshot(window_title: str = "", path: str = "", crop_to_map: bool = True, 
                       map_width_tiles: int = 16, map_height_tiles: int = 9) -> Image.Image | None:
    """
    Captures a screenshot of the game via API.
    
    Args:
        window_title: Ignored (kept for compatibility)
        path: Optional file path to save the screenshot.
        crop_to_map: Ignored (cropping handled by server)
        map_width_tiles: Ignored (kept for compatibility)
        map_height_tiles: Ignored (kept for compatibility)

    Returns:
        PIL Image object if successful, or None if failed.
    """
    print("Capturing screenshot via API...")
    
    try:
        client = get_api_client()
        image = client.get_game_screenshot()
        
        if image is None:
            print("Error: Failed to get screenshot from API")
            return None
            
        if path:
            image.save(path)
            print(f"Screenshot saved to: {path}")
            
        return image
        
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None


def send_key(key: str, window_title: str = 'Dungeon Escape AI') -> bool:
    """
    Send a keyboard action to the game via API.
    
    Args:
        key: The action/key to send
        window_title: Ignored (kept for compatibility)
        
    Returns:
        True if action was successful, False otherwise
    """
    print(f"Sending action '{key}' via API...")
    
    try:
        client = get_api_client()
        result = client.perform_action(key)
        
        if result is None:
            print(f"Error: Failed to perform action '{key}'")
            return False
            
        action_executed = result.get('action_executed', key)
        print(f"Action '{action_executed}' executed successfully")
        
        # Log state changes if available
        if 'state_changes' in result:
            state = result['state_changes']
            print(f"New state: Level {state.get('dungeon_level', '?')}, "
                  f"Step {state.get('current_level_step_count', '?')}, "
                  f"Health {state.get('player_health', '?')}")
        
        return True
        
    except Exception as e:
        print(f"Error sending action '{key}': {e}")
        return False


def get_game_state() -> dict | None:
    """
    Get current game state via API.
    
    Returns:
        Dictionary containing game state or None if error
    """
    try:
        client = get_api_client()
        return client.get_game_state()
    except Exception as e:
        print(f"Error getting game state: {e}")
        return None


def start_game(mode: str = "procedural", custom_map: str | None = None) -> dict | None:
    """
    Start a new game via API.
    
    Args:
        mode: Game mode - "custom", "procedural", or "string"
        custom_map: Map string for "string" mode
        
    Returns:
        Initial game state or None if error
    """
    try:
        client = get_api_client()
        return client.start_game(mode, custom_map)
    except Exception as e:
        print(f"Error starting game: {e}")
        return None


def start_game_from_file(map_file_path: str) -> dict | None:
    """
    Start a new game using a map from file.
    
    Args:
        map_file_path: Path to the map file
        
    Returns:
        Initial game state or None if error
    """
    try:
        client = get_api_client()
        return client.start_game_from_file(map_file_path)
    except Exception as e:
        print(f"Error starting game from file '{map_file_path}': {e}")
        return None


if __name__ == "__main__":
    # Test the API functions
    print("Testing Game API Client...")
    
    # Test connection
    client = get_api_client()
    if client.check_connection():
        print("✓ API server is accessible")
    else:
        print("✗ API server is not accessible")
        exit(1)
    
    # Test screenshot
    image = get_game_screenshot()
    if image:
        print("✓ Screenshot captured successfully")
    else:
        print("✗ Failed to capture screenshot")
    
    # Test game state
    state = get_game_state()
    if state:
        print(f"✓ Game state retrieved: {state}")
    else:
        print("✗ Failed to get game state")