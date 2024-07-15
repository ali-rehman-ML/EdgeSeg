# EdgeSeg

A library containing tools and optimized models for running semantic segmentation on edge devices with ARM CPUs.

## Utilities

### Memory Profiler

**Description:**
The Memory Profiler tool provides detailed profiling of PyTorch models, focusing on layer-by-layer execution time and CPU memory usage. It incorporates both high-level and low-level profiling information to analyze model performance.

**Table Content Explanation:**
- **Row ID:** Sequential identifier for each layer.
- **Layer:** Name of the layer in the PyTorch model.
- **Type:** Type of layer (e.g., Conv2d, BatchNorm2d).
- **Time (s):** Execution time of the layer during inference.
- **Memory (MB):** CPU memory usage of the layer.

### Usage

1. **Using Memory Profiler:**

   ```python
   from utils import ModelProfiler
   import torch
   import torchvision.models.segmentation as models

   # Define your model and input data
   model = models.deeplabv3_resnet101(pretrained=False)
   input_data = torch.randn(1, 3, 224, 224)

   # Create a profiler instance
   profiler = ModelProfiler(model)

   # Profile the model with input data
   profiler.profile(input_data)

   # Print detailed profiling information
   profiler.print_profiling_info()

   # Prompt user to input K for top K slowest layers
   k = int(input("Enter the value of K: "))
   
   # Print top K layers by execution time
   profiler.print_top_k_layers(k)
