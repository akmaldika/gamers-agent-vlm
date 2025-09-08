"""
Game API Client for Dungeon Escape AI

This module provides API-based interaction with the dungeon game server,
replacing direct window interaction with HTTP API calls.
"""

import requests
import json
import base64
from typing import Dict, Any, Optional, List
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class GameAPIClient:
    """Client for interacting with the Dungeon Game API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Game API Client.
        
        Args:
            base_url: Base URL of the game API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current game state.
        
        Returns:
            Dictionary containing game state or None if error
        """
        try:
            response = self.session.get(f"{self.base_url}/game-state")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error getting game state: {e}")
            return None
    
    def get_game_screenshot(self) -> Optional[Image.Image]:
        """
        Get current game screenshot. Requests raw PNG bytes (fmt=bytes) per API spec and
        crops to the map area if dimensions are provided via headers. Falls back to
        handling base64 JSON if server responds in b64 mode.

        Returns:
            PIL Image object (cropped to map area when possible) or None if error
        """
        try:
            # Prefer raw PNG bytes per spec (fmt=bytes). Server will include dimensions in headers.
            response = self.session.get(f"{self.base_url}/game-screenshot", params={"fmt": "bytes"})
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')

            if content_type.startswith('image/'):
                # Get tile and dimension information from headers
                tile_size = int(response.headers.get('X-Tile-Size', 16))
                total_width_tiles = int(response.headers.get('X-Total-Width-Tiles', 0))
                total_height_tiles = int(response.headers.get('X-Total-Height-Tiles', 0))
                map_width_tiles = int(response.headers.get('X-Map-Width-Tiles', 0))
                map_height_tiles = int(response.headers.get('X-Map-Height-Tiles', 0))
                total_width_pixels = int(response.headers.get('X-Total-Width-Pixels', 0))
                total_height_pixels = int(response.headers.get('X-Total-Height-Pixels', 0))
                map_width_pixels = int(response.headers.get('X-Map-Width-Pixels', 0))
                map_height_pixels = int(response.headers.get('X-Map-Height-Pixels', 0))

                # Convert binary data to PIL Image
                full_image = Image.open(BytesIO(response.content))
            
                # Calculate crop coordinates to extract only the map area
                # The map area starts from top-left corner (0,0)
                if map_width_pixels > 0 and map_height_pixels > 0:
                    # Map starts at top-left, so no offset needed
                    left = 0
                    top = 0
                    right = map_width_pixels
                    bottom = map_height_pixels

                    # Ensure crop coordinates are within image bounds
                    right = min(right, full_image.width)
                    bottom = min(bottom, full_image.height)

                    # Crop to map area only (from top-left)
                    cropped_image = full_image.crop((left, top, right, bottom))

                    # Store tile info in image info dictionary
                    cropped_image.info['tile_size'] = tile_size
                    cropped_image.info['width_tiles'] = map_width_tiles
                    cropped_image.info['height_tiles'] = map_height_tiles
                    cropped_image.info['total_width_tiles'] = total_width_tiles
                    cropped_image.info['total_height_tiles'] = total_height_tiles

                    return cropped_image
                else:
                    # Fallback: return full image if map dimensions not available
                    full_image.info['tile_size'] = tile_size
                    full_image.info['width_tiles'] = total_width_tiles
                    full_image.info['height_tiles'] = total_height_tiles
                    return full_image
            else:
                # Fallback if server returned JSON (fmt=b64)
                data = response.json()
                b64 = data.get("screenshot_png_base64")
                if not b64:
                    logger.error("Missing screenshot_png_base64 in response")
                    return None
                img_bytes = BytesIO(base64.b64decode(b64))
                img = Image.open(img_bytes)
                # Attach metadata when available
                for k_src, k_dst in [
                    ("tile_size", "tile_size"),
                    ("map_width_tiles", "width_tiles"),
                    ("map_height_tiles", "height_tiles"),
                    ("total_width_tiles", "total_width_tiles"),
                    ("total_height_tiles", "total_height_tiles"),
                ]:
                    if k_src in data:
                        img.info[k_dst] = data[k_src]
                return img
            
        except requests.RequestException as e:
            logger.error(f"Error getting game screenshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return None
    
    def start_game(self, mode: str = "procedural", custom_map: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Start a new game.
        
        Args:
            mode: Game mode - "procedural" or "string"
            custom_map: ASCII map string for "string" mode
            
        Returns:
            Initial game state or None if error
        """
        try:
            payload = {
                "mode": mode,
                "custom_map": custom_map
            }
            
            response = self.session.post(
                f"{self.base_url}/start-game",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error starting game: {e}")
            return None
    
    def perform_action(self, action: str) -> Optional[Dict[str, Any]]:
        """
        Perform an action in the game.
        
        Args:
            action: Action to perform (w, a, s, d, space, g, i, ., wait, etc.)
            
        Returns:
            Dictionary containing action result and new game state or None if error
        """
        try:
            payload = {"action": action}
            
            response = self.session.post(
                f"{self.base_url}/perform-action",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error performing action '{action}': {e}")
            return None
    
    def load_map_from_file(self, map_file_path: str) -> Optional[str]:
        """
        Load map content from a file.
        
        Args:
            map_file_path: Path to the map file
            
        Returns:
            Map content as string or None if error
        """
        try:
            with open(map_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading map from {map_file_path}: {e}")
            return None
    
    def start_game_from_file(self, map_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Start a new game using a map from file.
        
        Args:
            map_file_path: Path to the map file
            
        Returns:
            Initial game state or None if error
        """
        map_content = self.load_map_from_file(map_file_path)
        if map_content is None:
            return None
            
        return self.start_game(mode="string", custom_map=map_content)
    
    def check_connection(self) -> bool:
        """
        Check if the API server is accessible.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/game-state", timeout=5)
            # 200: OK; 400: No active game session per spec
            return response.status_code in (200, 400)
        except requests.RequestException:
            return False

# Convenience functions to maintain compatibility with existing code
_api_client = None

def get_api_client(base_url: str = "http://localhost:8000") -> GameAPIClient:
    """Get or create the global API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = GameAPIClient(base_url)
    return _api_client

def get_game_screenshot(**kwargs) -> Optional[Image.Image]:
    """Get game screenshot via API (compatibility function)."""
    client = get_api_client()
    return client.get_game_screenshot()

def send_key(key: str, **kwargs) -> bool:
    """Send key action via API (compatibility function)."""
    client = get_api_client()
    result = client.perform_action(key)
    return result is not None
