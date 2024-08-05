from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


def Segformer(name="b1"):
    if name=="b0":
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    elif name=="b1":
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    elif name=="b2":
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    elif name=="b3":
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    elif name=="b4":
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
    elif name=="b5":
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    else:
        print("Invalid Mdel Name ")