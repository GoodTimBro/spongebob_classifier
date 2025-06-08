import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load("mobilenet_spongebob.pth", map_location=device))
model.to(device)
model.eval()


classes = ["蟹老闆", "派大星", "皮老闆", "珊迪", "海綿寶寶", "章魚哥"]

def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]
    return {cls: float(probs[i]) for i, cls in enumerate(classes)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="海綿寶寶角色分類器",
    description="請上傳圖片，模型將預測圖片中的海綿寶寶角色。",
)

if __name__ == "__main__":
    iface.launch()
