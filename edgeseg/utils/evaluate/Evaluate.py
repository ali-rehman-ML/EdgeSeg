from edgeseg.utils.processor import out_process
import numpy as np
from edgeseg.utils.processor import EfficientVitImageProcessor,SegformerImageProcessor
import numpy as np
from PIL import Image



def plot_class_means(data):
    import matplotlib.pyplot as plt


    # Define the class names
    class_names = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle"
    ]

    # Create a dictionary mapping from index to class name
    index_to_class_name = {index: class_names[i] for i, index in enumerate(range(len(class_names)))}

    # Calculate the means for each class based on the `label_map`
    class_means = {}
    for i in range(len(class_names)):
            class_id=i
            class_mean = np.mean(data[:, i])
            class_means[class_id] = class_mean

    # Prepare data for plotting
    class_indices = class_means.keys()
    print(class_indices)
    means = [class_means[idx] for idx in class_indices]
    class_labels = [index_to_class_name[idx] for idx in class_indices]

    # Plotting the means as a bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, means)
    plt.xlabel('Class')
    plt.ylabel('Mean Value')
    plt.title('Mean Values of Each Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
label_map = np.array(
    (
        -1, -1, -1, -1, -1, -1, -1, 0,  # road 7
        1,  # sidewalk 8
        -1, -1, 2,  # building 11
        3,  # wall 12
        4,  # fence 13
        -1, -1, -1, 5,  # pole 17
        -1, 6,  # traffic light 19
        7,  # traffic sign 20
        8,  # vegetation 21
        9,  # terrain 22
        10,  # sky 23
        11,  # person 24
        12,  # rider 25
        13,  # car 26
        14,  # truck 27
        15,  # bus 28
        -1, -1, 16,  # train 31
        17,  # motorcycle 32
        18  # bicycle 33
    )
)

# Function to map the original image values to the new range
def map_image_values(image, label_map):
    if isinstance(image, Image.Image):
        image=np.array(image)


    mapped_image = np.copy(image)
    for original_value in range(len(label_map)):
        if label_map[original_value] != -1:
            mapped_image[image == original_value] = label_map[original_value]
        else:
            mapped_image[image == original_value] = 0  # Optionally map -1 values to 0
    return mapped_image





class Evaluate:
    def __init__(self,num_classes=19):
        self.a=0
        self.num_classes=19

    def evaluate_one(self,prediction,ground_truth):
        class_iou = {}
        ious = []

        ground_truth=map_image_values(ground_truth,label_map=label_map)
        for c in range(self.num_classes):
            # Create binary masks for the class
            gt_class = (ground_truth == c)
            pred_class = (prediction == c)

            # Calculate intersection and union
            intersection = np.logical_and(gt_class, pred_class).sum()
            union = np.logical_or(gt_class, pred_class).sum()

            if union == 0:
                # Handle case where class is not present in both ground truth and prediction
                iou = float('nan')
            else:
                iou = intersection / union

            class_iou[c] = iou
            if not np.isnan(iou):
                ious.append(iou)

        miou = np.nanmean(ious) if ious else float('nan')

        return class_iou, miou
    
    def evaluate_dataset(self,dataset,model,name='efficientvit',type='torch',device='cpu',input_size=512,samples=None,plot_class_analysis=True):
        IOU=[]
        Class_IOUS=[]
        if samples is None:
            samples=len(dataset)
        i=0
        for img,ground_truth in dataset:
            
            if name=='efficientvit':
                if type=='torch':
                    inp=EfficientVitImageProcessor.prepare_input(img,crop_size=input_size,type='torch')
                    inp.to(device)
                    model.to(device)
                    pred=model(inp)

                elif type=='ORT' or 'TFRT':
                    inp=EfficientVitImageProcessor.prepare_input(img,crop_size=input_size,type='numpy')
                    pred=model.invoke(inp)
                else:
                    print("Invalid model type")

            elif name=='segformer':
                if type=='torch':
                    inp=SegformerImageProcessor.prepare_input(img,crop_size=input_size,type='torch')
                    inp.to(device)
                    model.to(device)
                    pred=model(inp)

                elif type=='ORT' or 'TFRT':
                    inp=SegformerImageProcessor.prepare_input(img,crop_size=input_size,type='numpy')
                    pred=model.invoke(inp)
                else:
                    print("Invalid model type")

            else:
                print("Invalid Model Name")

            ciou,miou=self.evaluate_one(pred,ground_truth)
            IOU.append(miou)
            Class_IOUS.append(ciou)
            i=i+1
            if i==samples:
                break

        class_iou=np.array(Class_IOUS)

        if plot_class_analysis:
            plot_class_means(class_iou)


        return np.mean(IOU)

            




                    

                    
                


            
                    

                    

            




