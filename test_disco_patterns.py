import torch
import numpy as np
import matplotlib.pyplot as plt

def test_disco_patterns():
    """Test the disco pattern generation logic"""
    print("Testing disco pattern generation...")
    
    # Create test tensor shape (similar to latent space)
    batch_size, channels, height, width = 1, 4, 64, 64
    
    # Create coordinate grids
    y_coords = torch.linspace(-1, 1, height).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    x_coords = torch.linspace(-1, 1, width).view(1, 1, 1, width).expand(batch_size, 1, height, width)
    coords = torch.cat([y_coords, x_coords], dim=1)
    
    # Test pattern generation at different time steps
    for t in [0.0, 0.5, 1.0]:
        print(f"Generating patterns at time t={t}")
        
        # Generate disco patterns
        radial = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        angular = torch.atan2(coords[:, 0], coords[:, 1])
        
        patterns = (torch.sin(radial * 8 + t * 12) + 
                   torch.cos(angular * 6 + t * 10))
        
        # Create colorful patterns
        r_pattern = torch.sin(patterns * 1.0 + t * 6.28)
        g_pattern = torch.sin(patterns * 1.0 + t * 6.28 + 2.09)
        b_pattern = torch.sin(patterns * 1.0 + t * 6.28 + 4.19)
        
        color_patterns = torch.stack([r_pattern, g_pattern, b_pattern], dim=1)
        
        print(f"  Pattern range: [{patterns.min():.3f}, {patterns.max():.3f}]")
        print(f"  Color pattern shape: {color_patterns.shape}")
        
    print("Pattern generation test completed successfully!")

if __name__ == "__main__":
    test_disco_patterns()
