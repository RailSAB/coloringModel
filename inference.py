import os
import glob
import torch
from PIL import Image
import numpy as np
from models import build_models
from utils import lab_to_rgb
from torchvision import transforms
from imageio import imwrite

def load_image_as_L(path, size=(224, 224)):
    img = Image.open(path).convert('L').resize(size)
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape((1, 224, 224, 1))
    return arr

def find_latest_weight(path, pattern='*.pt'):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
        if not files:
            return None
        files = sorted(files, key=os.path.getmtime)
        return files[-1]
    elif os.path.isfile(path):
        return path
    return None

def infer_image(input_path, output_path, gen_weights, device=None, use_half=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[infer_image] device={device} half={use_half}')

    w = find_latest_weight(gen_weights)
    if w is None:
        raise ValueError(f'No weight file found at {gen_weights}')

    generator, _ = build_models(device=device, gen_weights=w)
    # Ensure model is on device explicitly:
    generator = generator.to(device)
    generator.eval()

    arr = load_image_as_L(input_path)
    to_tensor = transforms.ToTensor()
    L = to_tensor(arr[0]).unsqueeze(0).to(device=device, non_blocking=True).float()
    if use_half and device.type == 'cuda':
        generator.half()
        L = L.half()

    with torch.no_grad():
        pred_ab = generator(L)
    # Ensure pred_ab in float32 for lab_to_rgb if it expects that
    if pred_ab.dtype == torch.float16:
        pred_ab = pred_ab.float()

    rgb = lab_to_rgb(L.float(), pred_ab)
    imwrite(output_path, (rgb[0] * 255).astype('uint8'))
    if device.type == 'cuda':
        torch.cuda.synchronize()

def batch_infer(input_dir, output_dir, gen_weights, device=None, use_half=False):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*'))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print('No images found in', input_dir)
        return
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[batch_infer] device={device} half={use_half}')

    w = find_latest_weight(gen_weights)
    if w is None:
        raise ValueError(f'No weight file found at {gen_weights}')
    generator, _ = build_models(device=device, gen_weights=w)
    generator = generator.to(device)
    generator.eval()

    to_tensor = transforms.ToTensor()
    if use_half and device.type == 'cuda':
        generator.half()

    for f in files:
        arr = load_image_as_L(f)
        L = to_tensor(arr[0]).unsqueeze(0).to(device=device, non_blocking=True).float()
        if use_half and device.type == 'cuda':
            L = L.half()
        with torch.no_grad():
            pred_ab = generator(L)
        if pred_ab.dtype == torch.float16:
            pred_ab = pred_ab.float()
        rgb = lab_to_rgb(L.float(), pred_ab)
        out = os.path.join(output_dir, os.path.basename(f))
        imwrite(out, (rgb[0] * 255).astype('uint8'))
        print('Saved', out)
    if device.type == 'cuda':
        torch.cuda.synchronize()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--half', action='store_true', help='Use FP16 (CUDA only)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if os.path.isdir(args.input):
        batch_infer(args.input, args.output, args.weights, device=device, use_half=args.half)
    else:
        infer_image(args.input, args.output, args.weights, device=device, use_half=args.half)