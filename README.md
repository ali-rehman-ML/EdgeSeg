# EdgeSeg

A library containing tools and optimized models for running semantic segmentation on edge devices with ARM CPUs.

## Utilities

### 1-Memory Profiler

**Description:**
The Memory Profiler tool provides detailed profiling of PyTorch models, focusing on layer-by-layer execution time and CPU memory usage. It incorporates both high-level and low-level profiling information to analyze model performance.

**Output Table Content Explanation:**
- **Row ID:** Sequential identifier for each layer.
- **Layer:** Name of the layer in the PyTorch model.
- **Type:** Type of layer (e.g., Conv2d, BatchNorm2d).
- **Time (s):** Execution time of the layer during inference.
- **Memory (MB):** Current occupied  CPU memory.

### Usage

1. **Using Memory Profiler:**

   ```python
   from utils import ModelProfiler
   import torch
   import torchvision.models as models

   # Define your model and input data (be carefull make sure you use right input size for your model otherwise you may encounter error)
   model = models.resnet50(pretrained=True)
   model.cpu().eval()
   input_data=torch.randn(1, 3, 512, 512).cpu()

   # Create a profiler instance
   profiler = ModelProfiler(model)

   # Profile the model with input data
   profiler.profile(input_data)

   # Print detailed profiling information
   profiler.print_profiling_info()

   # Prompt user to input K for top K slowest layers
   k = 10
   
   # Print top K layers by execution time
   profiler.print_top_k_layers(k)

   ```
2.Numpy based Dataset and Dataset Loader 

