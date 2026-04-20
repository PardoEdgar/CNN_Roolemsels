import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
def build_model():
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
def test_visual(img_path, model_path=r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\xylem_unet.pth", threshold=0.5):
    """Muestra imagen original | máscara predicha | overlay superpuesto."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Carga y preprocesa
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    tf  = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    inp = tf(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

    # Devuelve al tamaño original
    prob_full = cv2.resize(prob, (w, h))
    mask      = (prob_full > threshold).astype(np.uint8)

    # Overlay: vasos en verde sobre la imagen
    overlay = img.copy()
    overlay[mask == 1] = [0, 220, 120]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);         axes[0].set_title("Original");         axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Máscara predicha"); axes[1].axis("off")
    axes[2].imshow(blended);     axes[2].set_title("Overlay");          axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    return mask

if __name__ == "__main__":

    # 2. Prueba visual sobre una imagen específica
    test_visual(
        img_path   = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\ROIs - Copy\Anatomia001.tiff",
        model_path = "xylem_unet.pth"
    )