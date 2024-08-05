# EdgeSeg

A library containing tools and optimized models for running semantic segmentation on edge devices with ARM CPUs.
## Model Zoo

### Usage

```python
from edgeseg.model_zoo import EfficientVit, Segformer , Deeplabv3

efficinetvit=EfficientVit(name="b1",weight_url='b1.pt')

segformer=Segformer(name="b1")

deeplabv3_mobv3=Deeplabv3()

```

#### For Efficinetvit download weight from [official Repo](https://github.com/mit-han-lab/efficientvit/blob/master/applications/seg.md#pretrained-models) , other 2 dont't require weight files.

### ONNX and TFlie Runtime inference 

#### Usage 

```python
from edgeseg.inference import ORT


model=ORT(model='efficientvit-b1.onnx')
output=model.invoke(input)

```


#### Download converted onnx and tflite models from [here](https://drive.google.com/drive/folders/1zFRNS8vf652Q50Ko-6oGOy-oLPewK7q_?usp=sharing)

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
- **Memory (MB):** Current occupied  GPU memory.
- **Input Shape:** Layer Input and Output shape.
- **Output Shape:** Layer Output and Output shape.



### Usage

**Using Memory Profiler:**

   ```python
   from edgeseg.utils import ModelProfiler
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
### 2. Dataset and Dataset Loader
   **Arguements**
   - **type:**      str : torchvsion or numpy.
   - **split:**     str : Dataset Split val or train.
   - **dir:**       str : Path to dataset directory wjere leftimg8bit and gtfine directories are.
   - **transform:** str : torchvision.transforms : transform to apply on Datset
   **Torchvision Dataset and Dataloader for pytorch model**
   ```python
   from edgeseg.utils.Datasets import Cityscapes
   dataset = Cityscapes(type='torchvision',split='val',dir='/cityscapes',transform=transforms)
   ```
#### Numpy Based Dataset and Dataloader
   The Numpy based Dataset and Dataloader make it easy to load Cityscapes Dataset in numpy having only numpy, PIL and cv2 as    dependendicies. Does not require pytorch and torchvision. It can be used for ONNX runtime and Tflite Runtime and             devices where pytorch is not supported.
   ```python
   from edgeseg.utils.Datasets import Cityscapes, Numpy_DataLoader
   
   dataset = Cityscapes(type='numpy',split='val',dir='/cityscapes',transform=None)
   val_loader = Numpy_DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

   #Usage

   for in in range(len(dataset)):
   image, ground_truth = dataset[i]

   #or using dataloader
   for images,ground_truth in val_loader:
      predictions=model(images)

   ```
**Numpy Transform Transforms**
- **Resize**
- **Random Crop**
- **Normalize**
- **To Tensor**
- **Random Horizontal Flip**
- **Random Rotation**
- **Color Jitter**
- **Grayscale**
#### Usage
```python
   from edgeseg.utils.Datasets import Cityscapes, Numpy_DataLoader
   from edgese.utils.transforms import normalize, to_tensor
   transforms = compose([
    to_tensor,
    lambda x: normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   
   dataset = Cityscapes(type='numpy',split='val',dir='/cityscapes',transform=transforms)

   ```
### 3. Image Pre Processing and Output post processing
To do inference we need to pre_process Image
```python
from edgeseg.utils.processor import EfficientVitImageProcessor, SegformerImageProcessor , out_process
from PIL import Image
import matplotlib.pyplt as plt
img=Image.open('image.png')
inp=EfficientVitImageProcessor(img,type='torch',crop_size=1024)
out=model(inp)

o=out_process.post_process_output(out,size=(1024,2048))
plt.imshow(o,cmap='gray')
plt.show()

```

### 4. Evaluateion
Evaluate One output or validation/testing dataset
```python
from edgeseg.utils.evaluate Evaluate

evaluator=Evaluate()


evaluator.evaluate_one(prediction,ground_truth)

#or evaulate over datset

evaluator.evaluate_dataset(dataset,model,name='efficientvit',type='torch',device='cpu',input_size=512,samples=None,plot_class_analysis=True):

#plot_class_analysis when true will plot the bar graph analysis ob individual classes miou over the dataset


```


## References and Resources
- **[EfficientVit](https://github.com/mit-han-lab/efficientvit)**
- **[Segformer](https://arxiv.org/abs/2105.15203)**
- **[Deeplabv3](https://arxiv.org/abs/1706.05587)**






   
    

