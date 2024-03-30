from dotenv import load_dotenv
import torch

load_dotenv('active.env')
torch.set_float32_matmul_precision('highest')
