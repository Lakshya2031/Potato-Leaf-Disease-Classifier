import argparse
import os
import torch
from torchvision import transforms
from PIL import Image

from train import build_model
from xai import GradCAM, get_default_target_layer, overlay_heatmap_on_image, vanilla_saliency, integrated_gradients


def load_checkpoint(model_path: str):
    ckpt = torch.load(model_path, map_location='cpu')
    arch = ckpt['arch']
    class_names = ckpt['class_names']
    model = build_model(arch, len(class_names), pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, class_names


def prepare_image(image_path: str, image_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return tf(img).unsqueeze(0)


def predict(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Predict potato leaf disease class')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--image', type=str, required=True, help='Path to a single image')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--explain', action='store_true', help='Generate Grad-CAM heatmap overlay next to prediction')
    parser.add_argument('--save-cam', type=str, default=None, help='Output path to save Grad-CAM image')
    parser.add_argument('--saliency', action='store_true', help='Also compute vanilla saliency and save if --save-cam provided')
    parser.add_argument('--ig', action='store_true', help='Also compute Integrated Gradients and save if --save-cam provided')
    args = parser.parse_args()

    model, class_names = load_checkpoint(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tensor = prepare_image(args.image, args.image_size)
    pred_idx, conf, prob_vector = predict(model, tensor, device)
    print(f"Prediction: {class_names[pred_idx]} (confidence {conf:.4f})")
    print("Probabilities:")
    for i, p in enumerate(prob_vector):
        print(f"  {class_names[i]}: {p:.4f}")

    if args.explain:
        # Grad-CAM
        try:
            target_layer = get_default_target_layer(model, arch=model.__class__.__name__.lower())
        except Exception:
            # fallback: try generic discovery
            target_layer = get_default_target_layer(model, arch='custom')
        cam = GradCAM(model, target_layer)
        heatmap, _ = cam.generate(tensor.to(device))
        cam.close()
        # load original image for overlay
        from PIL import Image
        orig = Image.open(args.image).convert('RGB')
        overlay = overlay_heatmap_on_image(orig, heatmap)
        if args.save_cam is not None:
            overlay.save(args.save_cam)
            print(f"Saved Grad-CAM overlay to {args.save_cam}")
        else:
            out_path = os.path.splitext(args.image)[0] + '_gradcam.jpg'
            overlay.save(out_path)
            print(f"Saved Grad-CAM overlay to {out_path}")
        if args.saliency:
            sal, _ = vanilla_saliency(model, tensor.to(device))
            # save saliency as image
            sal_img = Image.fromarray((sal*255).astype('uint8'))
            sal_path = (args.save_cam or out_path).replace('gradcam', 'saliency')
            sal_img.save(sal_path)
            print(f"Saved saliency map to {sal_path}")
        if args.ig:
            ig, _ = integrated_gradients(model, tensor.to(device))
            ig_img = Image.fromarray((ig*255).astype('uint8'))
            ig_path = (args.save_cam or out_path).replace('gradcam', 'integrated_gradients')
            ig_img.save(ig_path)
            print(f"Saved integrated gradients map to {ig_path}")

if __name__ == '__main__':
    main()
