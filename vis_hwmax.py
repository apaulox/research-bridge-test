import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import librosa
import sys

# Ensure imports work from the scripts absolute location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram, normalize

def process_image(image):
    image = image.resize((448, 224))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/unseen1_35', help='path to the specific checkpoint directory')
    parser.add_argument('--sample_ids', type=str, nargs='+', default=['123', '225', '817'], help='list of sample IDs')
    parser.add_argument('--audio_dir', type=str, default='/home/huskypaul/FAIR-Play/audios')
    parser.add_argument('--video_dir', type=str, default='/home/huskypaul/FAIR-Play/frames')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu to use')
    parser.add_argument('--audio_sampling_rate', type=int, default=16000)
    parser.add_argument('--audio_length', type=float, default=0.63)
    parser.add_argument('--output_name', type=str, default='hw_max_visualization.png', help='name of the output file')
    
    parser.add_argument('--visual_backbone', default='dinov3_vitb16')
    parser.add_argument('--dinov3_repo', default='/home/huskypaul/dinov3')
    parser.add_argument('--dinov3_weights', default='')
    opt = parser.parse_args()
    
    opt.sample_ids = [str(int(sid)).zfill(6) for sid in opt.sample_ids]
    
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
    else:
        device = torch.device('cpu')
        
    opt.weights_visual = os.path.join(opt.checkpoints_dir, 'mono2binaural', 'visual_best.pth')
    opt.weights_audio = os.path.join(opt.checkpoints_dir, 'mono2binaural', 'audio_best.pth')
    opt.unet_ngf = 64
    opt.unet_input_nc = 2
    opt.unet_output_nc = 2
    
    print(f"Loading weights from {opt.checkpoints_dir}...")
    builder = ModelBuilder()
    net_visual = builder.build_visual(
        weights=opt.weights_visual,
        backbone=opt.visual_backbone,
        dinov3_repo=opt.dinov3_repo,
        dinov3_weights=opt.dinov3_weights)
    net_audio = builder.build_audio(
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc,
            weights=opt.weights_audio)
    nets = (net_visual, net_audio)
    
    class DummyOpt:
        pass
    model_opt = DummyOpt()
    model_opt.device = device
    
    model = AudioVisualModel(nets, model_opt)
    
    if len(gpu_ids) > 0:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    model.eval()
    
    transform_list = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    vision_transform = transforms.Compose(transform_list)
    frames = []
    original_images = []
    audio_mix_specs = []
    audio_diff_specs = []
    valid_ids = []
    
    print(f"Extracting features for samples: {opt.sample_ids}...")
    
    for clip_id in opt.sample_ids:
        audio_path = os.path.join(opt.audio_dir, f"{int(clip_id)}.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(opt.audio_dir, f"{clip_id}.wav")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {audio_path} not found. Skipping...")
                continue
            
        audio, _ = librosa.load(audio_path, sr=opt.audio_sampling_rate, mono=False)
        audio_start_time = 0.0
        audio_end_time = audio_start_time + opt.audio_length
        audio_start = int(audio_start_time * opt.audio_sampling_rate)
        audio_end = audio_start + int(opt.audio_length * opt.audio_sampling_rate)
        audio_segment = audio[:, audio_start:audio_end]
        audio_segment = normalize(audio_segment)
        
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_segment[0, :] - audio_segment[1, :]))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_segment[0, :] + audio_segment[1, :]))
        
        frame_dir = os.path.join(opt.video_dir, clip_id)
        if not os.path.exists(frame_dir):
            frame_dir = os.path.join(opt.video_dir, str(int(clip_id)))
            
        frame_index = max(1, int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10)))
        frame_name = str(frame_index).zfill(6) + '.png'
        full_frame_path = os.path.join(frame_dir, frame_name)
        
        if not os.path.exists(full_frame_path):
             if not os.path.exists(frame_dir):
                 print(f"Warning: Directory {frame_dir} not found. Skipping...")
                 continue
             available_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
             if not available_frames:
                 print(f"Warning: No frames found for {clip_id}. Skipping...")
                 continue
             full_frame_path = os.path.join(frame_dir, available_frames[0])
        
        img = Image.open(full_frame_path).convert('RGB')
        img_proc = process_image(img)
        original_images.append(img_proc)
        frames.append(vision_transform(img_proc))
        
        audio_mix_specs.append(audio_mix_spec)
        audio_diff_specs.append(audio_diff_spec)
        valid_ids.append(clip_id)
        
    if not frames:
        print("Error: No valid data found.")
        return
        
    frames = torch.stack(frames).to(device)
    audio_mix_specs = torch.stack(audio_mix_specs).to(device)
    audio_diff_specs = torch.stack(audio_diff_specs).to(device)
    
    data = {
        'frame': frames,
        'audio_mix_spec': audio_mix_specs,
        'audio_diff_spec': audio_diff_specs
    }
    
    with torch.no_grad():
        output = model(data)
        
    import torch.nn.functional as F
    visual = output['visual_feat']
    B, D, H, W = visual.shape
    visual = F.normalize(visual.flatten(2), p=2, dim=1)
    # Count which HW location supplies each channel's max, before the dot product.
    # This is a channel-max diagnostic, not an audio-conditioned localization map.
    indices = visual.argmax(dim=2)
    vote_heatmaps = torch.stack([
        torch.bincount(row, minlength=H * W).float() for row in indices
    ]).reshape(B, H, W)

    # Plotting
    ncols = min(B, 4)
    nrows = (B + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
        
    for i in range(B):
        img_w, img_h = 448, 224
        patch_w = img_w / W
        patch_h = img_h / H
        
        ax = axes[i]
        ax.imshow(original_images[i])
        
        votes = vote_heatmaps[i].cpu().numpy() # [H, W]
        max_vote = votes.max()
        
        # Draw the patches and text
        for h_idx in range(H):
            for w_idx in range(W):
                v = votes[h_idx, w_idx]
                if v > 0:
                    center_x = (w_idx + 0.5) * patch_w
                    center_y = (h_idx + 0.5) * patch_h
                    
                    # Progressively red color based on vote count
                    alpha = 0.2 + 0.6 * (v / max_vote) if max_vote > 0 else 0.5
                    
                    # Colored patch
                    rect = plt.Rectangle((w_idx * patch_w, h_idx * patch_h), patch_w, patch_h, 
                                         linewidth=1, edgecolor='red', facecolor='red', alpha=alpha)
                    ax.add_patch(rect)
                    
                    # Write exact vote count
                    ax.text(center_x, center_y, str(int(v)), color='white', 
                            fontsize=8, ha='center', va='center', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.3, pad=0.5, edgecolor='none'))
        
        # We can still find the absolute max for the title
        max_idx_flat = votes.argmax()
        max_h, max_w = max_idx_flat // W, max_idx_flat % W
        
        ax.set_title(f"Sample: {valid_ids[i]}\nMax Vote: {int(max_vote)} at (h={max_h}, w={max_w})", fontsize=12)
        ax.axis('off')
        
    for i in range(B, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # Save logic
    ckpt_name = os.path.basename(os.path.normpath(opt.checkpoints_dir))
    mine_dir = os.path.dirname(os.path.abspath(__file__))
    hw_arg_dir = os.path.join(mine_dir, 'hw_arg', ckpt_name)
    os.makedirs(hw_arg_dir, exist_ok=True)
    
    # Determine the category of the first sample to use as base_name
    first_sample_id = int(valid_ids[0])
    category_file = os.path.join(mine_dir, 'data', 'category.json')
    base_name = "unknown"  # fallback
    
    if os.path.exists(category_file):
        import json
        with open(category_file, 'r') as f:
            categories = json.load(f)
            for cat, indices in categories.items():
                if first_sample_id in indices:
                    base_name = cat
                    break

    # Generate unique filename
    save_path = os.path.join(hw_arg_dir, f"{base_name}.png")
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(hw_arg_dir, f"{base_name}_{counter}.png")
        counter += 1
        
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    main()
