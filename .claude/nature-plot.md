# Nature Journal Plotting Standards

All figures in this project must comply with Nature journal formatting requirements.
**This file must be consulted before starting any plotting work.**

---

## 1. Figure Size & Resolution

- **Single-column figure**: width = 89 mm (3.5 in)
- **Double-column figure**: width = 183 mm (7.2 in)
- **Maximum height**: 247 mm (9.7 in)
- **Resolution**: minimum 300 dpi for raster outputs; prefer 600 dpi for line art
- **Output format**: PDF or SVG for vector; TIFF/PNG at ≥300 dpi for raster

```python
# Single-column
fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)

# Double-column
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=300)
```

---

## 2. Font

- **Typeface**: Arial or Helvetica (sans-serif)
- **Axis labels & tick labels**: 7–9 pt
- **Panel labels (a, b, c …)**: 8 pt, bold
- **Legend text**: 7–8 pt
- **No Times New Roman, no decorative fonts**

```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":      "Arial",
    "font.size":        8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.titlesize":   8,
    "figure.dpi":       300,
})
```

---

## 3. Color

### Approved palette

Use colorblind-safe, print-safe colors. Suggested scheme:

| Role | Hex | RGB |
|---|---|---|
| Blue (primary) | `#0072B2` | 0, 114, 178 |
| Orange | `#E69F00` | 230, 159, 0 |
| Green | `#009E73` | 0, 158, 115 |
| Red | `#D55E00` | 213, 94, 0 |
| Purple | `#CC79A7` | 204, 121, 167 |
| Sky blue | `#56B4E9` | 86, 180, 233 |
| Yellow | `#F0E442` | 240, 228, 66 |
| Black | `#000000` | — |

```python
NATURE_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]
```

- Do **not** use default matplotlib tab10 colors for publication.
- Continuous colormaps: prefer `viridis`, `plasma`, or `RdBu_r`. Avoid `jet`/`rainbow`.
- Heatmaps: use diverging colormap centered at zero when data spans negative/positive.
- All colors must be distinguishable in grayscale if the figure may be printed in black & white.

---

## 4. Line Weights & Marker Sizes

- **Data lines**: 1.0–1.5 pt
- **Axis spines & tick marks**: 0.5–0.8 pt
- **Error bars**: 0.8 pt, cap width 2 pt
- **Scatter markers**: 3–5 pt diameter (s=9–25 in matplotlib)

```python
mpl.rcParams.update({
    "lines.linewidth":    1.0,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})
```

---

## 5. Axes & Ticks

- Tick marks face **inward** (`direction='in'`).
- Show both x and y ticks on all four sides only when data density requires it; otherwise top/right spines off.
- Axis labels must include **units in parentheses**, e.g. `"Time (s)"`, `"Acceleration (m s⁻²)"`.
- Avoid redundant axis titles when the unit is already self-explanatory in context.
- Use `MaxNLocator` or explicit tick positions — avoid crowded automatic ticks.

```python
ax.tick_params(direction="in", top=True, right=True)
ax.spines[["top", "right"]].set_visible(False)   # if top/right spines unwanted
```

---

## 6. Legend

- Place inside the axes if space allows; outside only when the legend would overlap data.
- Use `frameon=False` or a lightweight frame (`edgecolor='none'`).
- Legend entries must match the order they appear in the figure.

```python
ax.legend(frameon=False, loc="upper right")
```

---

## 7. Error Representation

- Always state what error bars represent in the figure caption (SD, SEM, 95% CI, etc.).
- Preferred: mean ± SD for normally distributed data; median + IQR (box plot) otherwise.
- For n < 10 show individual data points overlaid on bars/boxes.

---

## 8. Statistical Annotations

- Mark significance brackets with asterisks: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant.
- Bracket lines: 0.6 pt, same color as axis spines.
- Report exact p values in the caption, not only the asterisk tier.

---

## 9. Panel Labels

- Lowercase bold: **a**, **b**, **c**, …
- Position: upper-left corner of each panel.
- Font: 8 pt, Arial Bold.

```python
ax.text(-0.15, 1.05, "a", transform=ax.transAxes,
        fontsize=8, fontweight="bold", va="top", ha="right")
```

---

## 10. Saving Figures

```python
fig.savefig("figure1.pdf", dpi=300, bbox_inches="tight", transparent=True)
fig.savefig("figure1.tiff", dpi=600, bbox_inches="tight")
```

- Always save as **both** PDF (for submission) and high-res TIFF/PNG (for review).
- Use `transparent=True` for PDF so figures can be composited on any background.
- Do **not** use `plt.show()` before `savefig()` — it resets the figure state in some backends.

---

## 11. Reusable rcParams Setup

Put this at the top of every plotting script or notebook:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

NATURE_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

mpl.rcParams.update({
    # Font
    "font.family":          "Arial",
    "font.size":            8,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "axes.titlesize":       8,
    # Lines & markers
    "lines.linewidth":      1.0,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    # Figure
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.transparent":  True,
    # Misc
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "legend.frameon":       False,
    "pdf.fonttype":         42,   # embed fonts as TrueType in PDF
    "ps.fonttype":          42,
})

plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=NATURE_COLORS)
```

---

## 12. Checklist Before Saving

- [ ] Figure width is 89 mm or 183 mm
- [ ] All fonts are Arial, ≥7 pt
- [ ] Colors are from the approved palette or a colorblind-safe colormap
- [ ] Tick marks point inward
- [ ] Axis labels include units
- [ ] Error bars are labeled in the caption
- [ ] Panel labels are lowercase bold (a, b, c …)
- [ ] Saved as PDF + TIFF/PNG at ≥300 dpi
- [ ] No `plt.show()` before `savefig()`
