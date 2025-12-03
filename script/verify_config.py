import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Add src to python path to allow imports if needed (though we just check config here)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def verify_config(cfg: DictConfig):
    print("Verifying Configuration Structure...")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Check for critical keys
    missing_keys = []
    
    # Check Client Config
    if not cfg.get("client"): missing_keys.append("client")
    else:
        if not cfg.client.get("provider"): missing_keys.append("client.provider")
        if not cfg.client.get("model"): missing_keys.append("client.model")
    
    # Check Agent Config
    if not cfg.get("agent"): missing_keys.append("agent")
    else:
        if not cfg.agent.get("max_loops"): missing_keys.append("agent.max_loops")
        if not cfg.agent.get("params"): missing_keys.append("agent.params")
    
    # Check Game Config
    if not cfg.get("game"): missing_keys.append("game")
    else:
        if not cfg.game.get("base_url"): missing_keys.append("game.base_url")
        
    if missing_keys:
        print("❌ Configuration Verification FAILED!")
        print("Missing keys:")
        for key in missing_keys:
            print(f"  - {key}")
        sys.exit(1)
    else:
        print("✅ Configuration Verification PASSED!")
        print("Structure appears correct.")

if __name__ == "__main__":
    verify_config()
