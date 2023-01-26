# import onnxruntime
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import numpy as np
from lightning_module import MuffinChihuahuaLightningModule
# def classify_muffin_chihuahua(img_path: str):
#     classes = ["Muffin", "Chihuahua"]
#     pil_img = Image.open(img_path).convert("RGB")
#     # transforms
#     transforms = Compose([Resize((224, 224)),
#                           ToTensor(),
#                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
#     transformed_img = transforms(pil_img)
#     ort_session = onnxruntime.InferenceSession("model.onnx")
#     input_name = ort_session.get_inputs()[0].name
#     ort_inputs = {input_name: transformed_img.numpy().reshape(1, 3, 224,224)}
#     ort_outs = ort_session.run(None, ort_inputs)
#     print(classes[np.argmax(ort_outs)])

def classify_muffin_chihuahua(img_path: str = None, pil_img: Image = None):
    classes = ["Muffin", "Chihuahua"]
    if img_path:
        pil_img = Image.open(img_path).convert("RGB")
    if pil_img:
        pil_img = pil_img.convert("RGB")
    # transforms
    transforms = Compose([Resize((224, 224)),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transformed_img = transforms(pil_img)
    transformed_img = transformed_img.view(1, 3, 224, 224)
    model = MuffinChihuahuaLightningModule().load_from_checkpoint('epoch=3-step=592.ckpt')
    model.eval()
    with torch.no_grad():
        y_hat = model(transformed_img)
        prediction = torch.argmax(y_hat)
        
    
    return classes[prediction]
    
print(classify_muffin_chihuahua("data/test/chihuhua_test_22.jpg"))