import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import r2_score

# Import your existing modules
from Config import my_POSSMConfig
from Model import my_POSSM
from Dataloader import my_dataset, pad_collate_fn

def get_inference_dataloader(data_path, batch_size=4, n_workers=0):
    """
    Creates a dataloader for the entire dataset without splitting 
    (since we are doing inference on a distinct session).
    """
    if not os.path.exists(data_path):
        print(f"Warning: File not found {data_path}")
        return None
    
    dataset = my_dataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle for inference
        num_workers=n_workers,
        drop_last=False, # Keep all data
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return loader

def calculate_r2(y_true, y_pred):
    """
    Calculate R2 score for 2D velocity.
    Returns the average R2 across X and Y dimensions.
    """
    # y_true, y_pred shape: (N_samples, 2)
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    return (r2_x + r2_y) / 2, r2_x, r2_y

@torch.no_grad()
def evaluate_session(model, loader, device, config, train_mean, train_std):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # Validation loop
    for spike, bin_mask, spike_mask, vel, vel_lens in loader:
        spike = spike.to(device)
        bin_mask = bin_mask.to(device)
        spike_mask = spike_mask.to(device)
        vel = vel.to(device)
        vel_lens = vel_lens.to(device)
        
        # Determine valid output length based on k_history lag
        # Logic matches train_one_epoch in long_term_main.py
        effective_lens = vel_lens - (config.k_history - 1) * config.bin_size
        max_valid_time = effective_lens.max()
        
        # 1. Forward Pass
        # outputs shape: (batch, ~max_time, 2) - Normalized Space
        outputs = model(spike, bin_mask, spike_mask)
        
        # Clip to valid time length
        outputs = outputs[:, :max_valid_time, :]
        
        # 2. Process Batch
        for i in range(len(vel)):
            curr_len = effective_lens[i].item()
            if curr_len <= 0:
                continue
            
            # --- Extract Prediction ---
            # The model predicts normalized velocity based on Session 0 stats
            pred_norm = outputs[i, :curr_len, :] 
            
            # Denormalize using TRAINING (Session 0) stats
            # We want to see if the model's logic holds up in physical space
            pred_phys = (pred_norm * train_std) + train_mean
            
            # --- Extract Ground Truth ---
            # Ground truth needs to be aligned. 
            # In training, we compared outputs against: vel[:, (k-1)*bin_size:, :]
            start_idx = (config.k_history - 1) * config.bin_size
            target_phys = vel[i, start_idx : start_idx + curr_len, :]
            
            all_preds.append(pred_phys.cpu().numpy())
            all_targets.append(target_phys.cpu().numpy())

    if not all_preds:
        return None

    # Concatenate all trials to calculate global R2 for this session
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    
    # Calculate Metrics
    mse = np.mean((all_preds_np - all_targets_np) ** 2)
    avg_r2, r2_x, r2_y = calculate_r2(all_targets_np, all_preds_np)
    
    return {
        "mse": mse,
        "avg_r2": avg_r2,
        "r2_x": r2_x,
        "r2_y": r2_y
    }

def main():
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = my_POSSMConfig()
    model_path = "./long_term_model.pth"
    
    # --- 2. Load Model ---
    print(f"Loading model from {model_path}...")
    model = my_POSSM(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # --- 3. Load Session 0 Metadata (Training Distribution) ---
    # We must use Session 0's Mean/Std to denormalize predictions 
    # because the model weights are fixed to that output distribution.
    meta_path_0 = "long_term_data/Chewie_processed/session_0/meta_data.json"
    with open(meta_path_0, "r") as f:
        meta_0 = json.load(f)
    
    # These tensors are used to denormalize the model output
    train_mean = torch.tensor(meta_0["vel_mean"], device=device, dtype=torch.float32)
    train_std = torch.tensor(meta_0["vel_std"], device=device, dtype=torch.float32)
    
    print(f"Training Baseline (Session 0) Loaded.")
    print("-" * 50)

    # --- 4. Inference Loop (Session 1 to 11) ---
    # We can also include Session 0 to verify training performance
    results_summary = []
    
    for session_id in range(12): # 0 to 11
        data_path = f"long_term_data/Chewie_processed/session_{session_id}/sliced_trials.pt"
        
        loader = get_inference_dataloader(data_path)
        
        if loader is None:
            continue
            
        print(f"Evaluating Session {session_id}...", end=" ")
        
        metrics = evaluate_session(model, loader, device, config, train_mean, train_std)
        
        if metrics:
            print(f"R2: {metrics['avg_r2']:.4f} (MSE: {metrics['mse']:.4f})")
            results_summary.append({
                "session": session_id,
                **metrics
            })
        else:
            print("Failed (No valid data)")

    # --- 5. Final Report ---
    print("\n" + "="*50)
    print(f"{'Session':<10} | {'R2 (Avg)':<10} | {'R2 (X)':<10} | {'R2 (Y)':<10} | {'MSE':<10}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['session']:<10} | {res['avg_r2']:<10.4f} | {res['r2_x']:<10.4f} | {res['r2_y']:<10.4f} | {res['mse']:<10.4f}")
    print("="*50)

if __name__ == "__main__":
    main()