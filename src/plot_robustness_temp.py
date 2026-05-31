import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for rich premium aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Define rotation angles
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
columns = [f"{a}deg" for a in angles]

# Specific failure rate data from user (None represents '---')
data = {
    'A': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'B': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'C': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.5, 34.3],
    'D': [None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.4],
    'E': [0.0, 1.5, 12.7, 3.1, 1.6, 10.3, 62.2, 96.0, 100.0, 100.0, 100.0, 100.0],
    'F': [0.0, 0.0, 0.0, 0.0, 0.0, 3.8, 4.1, 1.7, 7.9, 13.9, 23.4, 58.5],
    'G': [None, None, None, None, None, None, 25.0, 70.7, 75.5, 25.9, 24.3, 16.7],
    'H': [100.0, 100.0, 100.0, 100.0, None, 100.0, 61.5, 22.0, 25.2, 10.6, 16.7, None],
    'I': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.7],
    'L': [0.0, 0.0, 0.0, 1.2, 30.9, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    'Y': [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 2.2, 44.4, 69.4, 100.0]
}

df = pd.DataFrame.from_dict(data, orient='index', columns=columns)

# Create models folder
output_dir = "models/rotation_analysis"
os.makedirs(output_dir, exist_ok=True)

# ----------------- PLOT: SMALL MULTIPLES GRID (Highly readable!) -----------------
# 4 rows, 3 columns to house all 11 labels + 1 legend/summary subplot
fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=False, sharey=False, dpi=300)
axes_flat = axes.flatten()

labels_sorted = sorted(df.index)

# Unified styling parameters
line_color = '#2980b9'
fill_color = '#3498db'

# Loop through sorted labels to draw each grid cell
for idx, label in enumerate(labels_sorted):
    ax = axes_flat[idx]
    row = df.loc[label]
    valid_mask = row.notna()
    
    x_vals = np.array(angles)[valid_mask]
    y_vals = row[valid_mask].values.astype(float)
        
    # Plot line & fill area under it
    ax.plot(x_vals, y_vals, color=line_color, linewidth=2.5, marker='o', markersize=4, label='Failure Rate')
    ax.fill_between(x_vals, y_vals, color=fill_color, alpha=0.15)
    
    # Subplot details
    ax.set_title(f"Label {label}", fontsize=14, fontweight='bold', pad=8, color='#2c3e50')
    ax.set_xlim(-5, 115)
    ax.set_ylim(-5, 105)
    
    # Apply X ticks (Degrees) and Y ticks (Failure rate %) to every single chart
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a}°" for a in angles], rotation=45, fontsize=8)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)], fontsize=8)
    
    ax.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')

# Use 12th subplot for color legend / explanation
ax_legend = axes_flat[-1]
ax_legend.axis('off')
ax_legend.text(0.1, 0.45, "Grey gaps (---) represent\nno recorded dataset sample.", color='#7f8c8d', fontsize=11, style='italic')

plt.suptitle("ASL SVM Rotation Robustness - Small Multiples Facet Grid", fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
plt.tight_layout(rect=[0, 0, 1, 0.96])

grid_chart_path = os.path.join(output_dir, "robustness_comparison_grid.png")
plt.savefig(grid_chart_path, bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("SUCCESSFULLY PLOTTED HIGH-QUALITY VISUALIZATIONS!")
print("="*50)
print(f"Grid Facet Chart saved to:  {os.path.abspath(grid_chart_path)}")
print("="*50 + "\n")
