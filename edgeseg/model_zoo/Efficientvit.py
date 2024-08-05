from .efficientvit.seg_model_zoo import create_seg_model

def EfficientVit(name="b1",weight_url=None):
    
    model = create_seg_model(
    name=name, dataset="cityscapes", weight_url=weight_url,pretrained=True
    )
    return model

