import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_hourly_figures import fig05_rmse_vs_horizon
fig05_rmse_vs_horizon()
print("done")
