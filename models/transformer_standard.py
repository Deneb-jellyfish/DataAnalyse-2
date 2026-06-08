"""Standard Transformer for PM2.5 prediction (quick comparison vs Informer).

Uses PyTorch's built-in nn.MultiheadAttention with full attention —
more suitable than ProbSparse for short sequences (seq_len=30).
"""

from __future__ import annotations

import logging, math, sys
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.io import PROCESSED_DIR

log = logging.getLogger(__name__)

# config
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
SEQ_LEN, LABEL_LEN, PRED_LEN = 30, 15, 1
TEST_RATIO = 0.2
D_MODEL, N_HEADS, D_FF = 128, 4, 512
E_LAYERS, D_LAYERS = 2, 1
DROPOUT = 0.2
BATCH_SIZE, LR, EPOCHS, PATIENCE = 32, 5e-4, 200, 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = ["PM2.5","PM10","SO2","NO2","CO","O3",
                "TEMP","PRES","DEWP","RAIN","WSPM"]
TARGET_IDX = 0

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.embed = nn.Linear(c_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.pos(self.embed(x)))

# ---- Transformer Encoder ----
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(a))
        return self.norm2(x + self.dropout(self.ff(x)))

# ---- Transformer Decoder ----
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(a))
        a, _ = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout(a))
        return self.norm3(x + self.dropout(self.ff(x)))

# ---- Full Model ----
class TransformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out=1, seq_len=SEQ_LEN,
                 label_len=LABEL_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.enc_embed = DataEmbedding(enc_in)
        self.dec_embed = DataEmbedding(dec_in)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(E_LAYERS)])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(D_LAYERS)])
        self.proj = nn.Linear(D_MODEL, c_out)
    def forward(self, x_enc, x_dec):
        enc = self.enc_embed(x_enc)
        for layer in self.encoder: enc = layer(enc)
        dec = self.dec_embed(x_dec)
        for layer in self.decoder: dec = layer(dec, enc)
        return self.proj(dec)[:, -self.pred_len:, :]

# ---- Data pipeline (same as Informer) ----
def _load_data():
    df = pd.read_csv(PROCESSED_DIR/"beijing"/"beijing_daily_city.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    raw = df[FEATURE_COLS].values.astype(np.float64)
    raw = np.where(np.isnan(raw), np.nanmean(raw,axis=0), raw)
    scaler = StandardScaler(); scaled = scaler.fit_transform(raw)
    return df, scaled, scaler

def _build_windows(data, seq_len, label_len, pred_len):
    enc_in, dec_in, targets = [], [], []
    for i in range(len(data)-seq_len-pred_len+1):
        enc_in.append(data[i:i+seq_len])
        d = np.zeros((label_len+pred_len, data.shape[1]), dtype=np.float32)
        d[:label_len] = data[i+seq_len-label_len:i+seq_len]
        dec_in.append(d)
        targets.append(data[i+seq_len, TARGET_IDX])
    return (np.stack(enc_in).astype(np.float32),
            np.stack(dec_in).astype(np.float32),
            np.array(targets,dtype=np.float32)[:,np.newaxis])

def _split(*arrs):
    n=len(arrs[0]); s=int(n*(1-TEST_RATIO))
    return [(a[:s],a[s:]) for a in arrs]

def _loader(xe,xd,y,bs,shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(xe),torch.from_numpy(xd),
                                    torch.from_numpy(y)), batch_size=bs, shuffle=shuffle)

# ---- Run ----
def run():
    df, scaled, scaler = _load_data()
    x_enc, x_dec, y = _build_windows(scaled, SEQ_LEN, LABEL_LEN, PRED_LEN)
    (xe_tr, xe_te), (xd_tr, xd_te), (y_tr, y_te) = _split(x_enc, x_dec, y)

    pm25_mean = scaler.mean_[TARGET_IDX]
    pm25_scale = scaler.scale_[TARGET_IDX]
    y_test_orig = y_te[:,0] * pm25_scale + pm25_mean

    log.info("Transformer: train=%d test=%d seq_len=%d", len(xe_tr), len(xe_te), SEQ_LEN)
    log.info("  d_model=%d heads=%d params=%.0fK",
             D_MODEL, N_HEADS,
             sum(p.numel() for p in TransformerModel(11,11).parameters())/1000)

    vs = int(len(xe_tr)*0.15)
    tr_ld = _loader(xe_tr[:-vs],xd_tr[:-vs],y_tr[:-vs],BATCH_SIZE,False)
    va_ld = _loader(xe_tr[-vs:],xd_tr[-vs:],y_tr[-vs:],BATCH_SIZE,False)

    model = TransformerModel(11, 11).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    best_val, pat, best_st = float("inf"), PATIENCE, None

    for ep in range(EPOCHS):
        model.train(); tl=0.0
        for Xe,Xd,Yb in tr_ld:
            Xe,Xd,Yb = Xe.to(DEVICE),Xd.to(DEVICE),Yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xe,Xd).squeeze(-1), Yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tl += loss.item()*Xe.size(0)
        tl/=len(tr_ld.dataset)
        model.eval(); vl=0.0
        with torch.no_grad():
            for Xe,Xd,Yb in va_ld:
                Xe,Xd,Yb = Xe.to(DEVICE),Xd.to(DEVICE),Yb.to(DEVICE)
                vl += crit(model(Xe,Xd).squeeze(-1),Yb).item()*Xe.size(0)
        vl/=len(va_ld.dataset)
        if vl < best_val: best_val=vl; pat=PATIENCE; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            pat-=1
            if pat==0: log.info("  early stop epoch %d",ep+1); break
    model.load_state_dict(best_st)

    model.eval()
    with torch.no_grad():
        preds=[]
        for Xe,Xd,_ in _loader(xe_te,xd_te,y_te,BATCH_SIZE,False):
            Xe,Xd = Xe.to(DEVICE),Xd.to(DEVICE)
            preds.append(model(Xe,Xd).cpu().numpy())
    y_pred = np.concatenate(preds).ravel() * pm25_scale + pm25_mean

    residuals = y_test_orig - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ape = np.abs(residuals/np.maximum(y_test_orig,1e-6))*100
    mape = float(np.mean(np.minimum(ape,200.0)))

    offset = SEQ_LEN + int(len(x_enc)*(1-TEST_RATIO))
    dates = df["date"].iloc[offset:offset+len(y_pred)].values
    rows = []
    for d,t,p in zip(dates,y_test_orig,y_pred):
        rows.append({"date":pd.Timestamp(d).strftime("%Y-%m-%d"),"city":"Beijing",
                      "model":"Transformer","y_true":round(float(t),4),
                      "y_pred":round(max(0.0,float(p)),4)})

    log.info("Transformer  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%",rmse,mae,mape)
    return {"model":"Transformer","rmse":rmse,"mae":mae,"mape":mape,
            "n_train":len(xe_tr),"n_test":len(xe_te),"predictions":rows}

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    r=run()
    print(f"\n{'='*50}")
    print(f"Transformer | RMSE={r['rmse']:.2f} MAE={r['mae']:.2f} MAPE={r['mape']:.1f}% (n={r['n_test']})")
