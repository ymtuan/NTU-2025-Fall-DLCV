import torch
from model import UNet

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
ckpt = torch.load('../../hw2_data/face/UNet.pt', map_location=device)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()

# Create dummy inputs
x = torch.randn(1, 3, 64, 64, device=device)

# Test with different timestep formats
with torch.no_grad():
    # Test 1: 0-indexed timestep
    t1 = torch.tensor([500], device=device, dtype=torch.long)
    out1 = model(x, t1)
    print(f"t=500 (0-idx): output range [{out1.min():.6f}, {out1.max():.6f}]")
    
    # Test 2: 1-indexed timestep  
    t2 = torch.tensor([501], device=device, dtype=torch.long)
    out2 = model(x, t2)
    print(f"t=501 (1-idx): output range [{out2.min():.6f}, {out2.max():.6f}]")
    
    # Test 3: very high value
    t3 = torch.tensor([999], device=device, dtype=torch.long)
    out3 = model(x, t3)
    print(f"t=999: output range [{out3.min():.6f}, {out3.max():.6f}]")
    
    # Test 4: very low value
    t4 = torch.tensor([0], device=device, dtype=torch.long)
    out4 = model(x, t4)
    print(f"t=0: output range [{out4.min():.6f}, {out4.max():.6f}]")
    
    # Test 5: value of 1
    t5 = torch.tensor([1], device=device, dtype=torch.long)
    out5 = model(x, t5)
    print(f"t=1: output range [{out5.min():.6f}, {out5.max():.6f}]")
