import json
import matplotlib.pyplot as plt
from pathlib import Path

waypoints_path = Path(__file__).resolve().parent.parent / "config" / "tracks" / "default" / "waypoints.json"

with open(waypoints_path) as f:
    data = json.load(f)

wps = data["waypoints"]
xs = [p[0] for p in wps]
ys = [p[1] for p in wps]
zs = [p[2] for p in wps]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].plot(xs, zs, linewidth=0.8)
axes[0].plot(xs[0], zs[0], "go", markersize=10, label="Start")
axes[0].plot(xs[-1], zs[-1], "r*", markersize=10, label="End")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")
axes[0].set_title(f"Track layout (X vs Z) - {data['track_name']}")
axes[0].legend()
axes[0].set_aspect("equal")
axes[0].grid(True, alpha=0.3)

for i in range(0, len(wps), max(1, len(wps) // 20)):
    axes[0].annotate(str(i), (xs[i], zs[i]), fontsize=6, alpha=0.7)

axes[1].plot(range(len(wps)), ys, linewidth=0.8)
axes[1].set_xlabel("Waypoint index")
axes[1].set_ylabel("Y (height)")
axes[1].set_title("Elevation profile")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(waypoints_path.parent / "waypoints_plot.png", dpi=150)
print(f"Plot saved to {waypoints_path.parent / 'waypoints_plot.png'}")
print("Done!")