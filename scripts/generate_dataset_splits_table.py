#!/usr/bin/env python3
"""
Generate publication-grade dataset split count table.
Matches academic report formatting with sequence-stratified split statistics.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("docs/reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def generate_table_image():
    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=300, facecolor='#FFFFFF')
    ax.axis('off')

    headers = ['Dataset', 'Training images', 'Validation images', 'Test images']
    rows = [
        ['Anti-UAV', '14,513', '6,721', '3,570'],
        ['Det-Fly', '6,286', '—', '4,384'],
        ['DUT-Anti-UAV', '9,748', '—', '—'],
        ['Additional negative samples', '686', '100', '98'],
        ['Total', '31,233', '6,821', '8,052']
    ]

    col_widths = [0.42, 0.20, 0.20, 0.18]

    table = ax.table(cellText=rows, colLabels=headers, colWidths=col_widths, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('#FFFFFF')
        cell.set_linewidth(0)

        if col == 0:
            cell.set_text_props(ha='left')
        else:
            cell.set_text_props(ha='center')

        if row == 0:
            cell.set_text_props(weight='bold', color='#0F172A')
            cell.visible_edges = 'B'
            cell.set_edgecolor('#E2E8F0')
            cell.set_linewidth(1.2)
        elif row == len(rows):
            cell.set_text_props(weight='bold', color='#0F172A')
            cell.visible_edges = 'T'
            cell.set_edgecolor('#E2E8F0')
            cell.set_linewidth(1.2)
        else:
            cell.set_text_props(color='#334155')
            cell.visible_edges = 'B'
            cell.set_edgecolor('#F1F5F9')
            cell.set_linewidth(0.8)

    out_p = OUT_DIR / 'dataset_splits_table.png'
    plt.savefig(out_p, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_p}")


if __name__ == '__main__':
    generate_table_image()
