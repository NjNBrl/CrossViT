import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (torch):", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
