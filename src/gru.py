"""
src/gru.py - Step 4: Unsupervised GRU Autoencoder
Extracts 11-month sequence changes, trains the bottleneck on pre-2019 data using AdamW, 
and generates latent embeddings.
"""

import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.config import DATA_DIR, CACHE_DIR

class SequenceGRUAutoencoder(nn.Module):
    def __init__(self, input_dim=147, hidden_dim=64, latent_dim=16):
        super().__init__()
        # Encoder: 1-layer GRU to 64 hidden, then linear to latent d
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: GRU reconstructs sequence from the latent vector
        self.decoder_gru = nn.GRU(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch, 11, 147)
        _, hidden = self.encoder_gru(x)
        z = self.encoder_linear(hidden.squeeze(0))  # Latent embedding 'd'
        
        # Repeat embedding across the 11-month sequence length for the decoder
        seq_len = x.size(1)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder_gru(z_repeated)
        reconstruction = self.decoder_linear(dec_out)
        
        return reconstruction, z

def build_sequences(df: pl.DataFrame, factors: list) -> pl.DataFrame:
    """
    Computes 11 month-to-month changes over t-11..t and extracts sequences.
    Requires at least 6 valid months; otherwise, marked as invalid.
    """
    # Sort chronologically per stock
    df = df.sort(["permno", "eom"])
    
    # Calculate month-to-month changes
    diff_exprs = [pl.col(f).diff().alias(f"{f}_diff") for f in factors]
    df = df.with_columns(diff_exprs)
    
    diff_cols = [f"{f}_diff" for f in factors]
    
    # Fast rolling sequence extraction using Pandas/NumPy
    # Polars rolling over multiple columns is currently limited, so we bridge to NumPy
    pdf = df.select(["permno", "eom"] + diff_cols).to_pandas()
    
    sequences = []
    valid_masks = []
    
    for _, group in pdf.groupby("permno"):
        vals = group[diff_cols].values
        eoms = group["eom"].values
        
        # Need at least 11 rows to form one complete t-11..t window
        if len(vals) < 11:
            seqs = np.zeros((len(vals), 11, len(factors)))
            valid = np.zeros(len(vals), dtype=bool)
        else:
            # Create sliding windows of shape (N-10, 11, 147)
            window_view = np.lib.stride_tricks.sliding_window_view(vals, window_shape=(11, len(factors)))
            window_view = window_view.squeeze(1) # Drop dummy dimension
            
            # Pad the first 10 rows with zeros (invalid)
            pad = np.zeros((10, 11, len(factors)))
            seqs = np.concatenate([pad, window_view], axis=0)
            
            # Require at least 6 valid months (non-NaN rows per sequence)
            # A row is valid if it doesn't contain NaNs (or checking all features)
            valid_counts = (~np.isnan(seqs).any(axis=2)).sum(axis=1)
            valid = valid_counts >= 6
            
        sequences.extend(seqs)
        valid_masks.extend(valid)
        
    df = df.with_columns([
        pl.Series("sequence_valid", valid_masks)
    ])
    
    # Convert sequence list to a stacked NumPy array for PyTorch
    stacked_seqs = np.stack(sequences)
    
    # Replace any remaining NaNs in valid sequences with 0 for the PyTorch MSE
    stacked_seqs = np.nan_to_num(stacked_seqs, nan=0.0)
    
    return df, stacked_seqs


def train_and_embed(d: int, df: pl.DataFrame, factors: list, device="cuda"):
    """
    Trains the GRU Autoencoder and returns the embeddings.
    Weight decay is tuned dynamically based on 2019-2020 validation loss.
    """
    df, seq_array = build_sequences(df, factors)
    
    # Define splits
    train_mask = (df["eom"] <= pl.date(2018, 11, 30)) & df["sequence_valid"]
    val_mask = (df["eom"] >= pl.date(2019, 1, 1)) & (df["eom"] <= pl.date(2020, 12, 31)) & df["sequence_valid"]
    
    train_idx = df.filter(train_mask).select(pl.arange(0, len(df))).to_series().to_list()
    val_idx = df.filter(val_mask).select(pl.arange(0, len(df))).to_series().to_list()
    
    X_train = torch.tensor(seq_array[train_idx], dtype=torch.float32)
    X_val = torch.tensor(seq_array[val_idx], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=256, shuffle=False)
    
    best_overall_val_loss = float('inf')
    best_model_state = None
    best_wd = 0
    
    # Grid search weight decay by 2019-2020 reconstruction loss
    for wd in [0, 1e-4, 1e-3]:
        model = SequenceGRUAutoencoder(input_dim=147, hidden_dim=64, latent_dim=d).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        criterion = nn.MSELoss()
        
        patience, best_val_loss = 5, float('inf')
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                
                # Clip gradient norm at 1.0 on every step
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Validation Step
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon, _ = model(x)
                    val_loss += criterion(recon, x).item()
                    
            val_loss /= len(val_loader)
            print(f"WD: {wd} | Epoch: {epoch} | Val MSE: {val_loss:.4f}")
            
            # Early stopping on 2019-2020 reconstruction loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if val_loss < best_overall_val_loss:
                    best_overall_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_wd = wd
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
                    
    print(f"GRU (d={d}) selected weight_decay={best_wd} with Val MSE: {best_overall_val_loss:.4f}")
    
    # Generate final embeddings for the entire dataset
    final_model = SequenceGRUAutoencoder(input_dim=147, hidden_dim=64, latent_dim=d).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()
    
    all_loader = DataLoader(TensorDataset(torch.tensor(seq_array, dtype=torch.float32)), batch_size=512, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in all_loader:
            x = batch[0].to(device)
            _, z = final_model(x)
            embeddings.append(z.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Nullify embeddings that didn't meet the 6-month validity rule
    invalid_mask = ~df["sequence_valid"].to_numpy()
    embeddings[invalid_mask] = np.nan
    
    # Append gru_1..gru_d to DataFrame
    gru_cols = [f"gru_{i+1}" for i in range(d)]
    df_out = df.select(["permno", "eom"])
    for i, col in enumerate(gru_cols):
        df_out = df_out.with_columns(pl.Series(col, embeddings[:, i]))
        
    return df_out