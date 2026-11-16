"""Export measurements and histograms from the edited plaque labels."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def measure_plaques(labels):
    """Return plaque IDs, areas in pixels, and second-moment aspect ratios."""
    rows, cols = np.nonzero(labels)
    ids, groups, areas = np.unique(
        labels[rows, cols], return_inverse=True, return_counts=True,
    )
    if not ids.size:
        return ids, areas, np.empty(0)

    # Dense group indices keep memory independent of the largest plaque ID.
    row_offsets = rows - (np.bincount(groups, weights=rows) / areas)[groups]
    col_offsets = cols - (np.bincount(groups, weights=cols) / areas)[groups]
    covariance = np.empty((ids.size, 2, 2))
    covariance[:, 0, 0] = np.bincount(groups, weights=row_offsets**2) / areas
    covariance[:, 1, 1] = np.bincount(groups, weights=col_offsets**2) / areas
    covariance[:, 1, 0] = covariance[:, 0, 1] = (
        np.bincount(groups, weights=row_offsets * col_offsets) / areas
    )
    minor, major = np.linalg.eigvalsh(covariance).T
    # Collinear pixels have a zero minor eigenvalue, up to rounding error.
    defined = minor > 8 * np.finfo(float).eps * major
    ratios = np.full(ids.size, np.nan)
    ratios[defined] = np.sqrt(major[defined] / minor[defined])
    return ids, areas, ratios


def save_histogram(destination, values, xlabel, empty_message, *, total_count):
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    with rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": 16,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2.5,
        "legend.frameon": False,
    }):
        figure = Figure(figsize=(6, 4), layout="constrained", facecolor="white")
        FigureCanvasAgg(figure)
        title = f"Total plaques: {total_count:,}"
        if values.size < total_count:
            title += f" ({values.size:,} plotted)"
        figure.suptitle(title, fontsize=13)
        axes = figure.subplots()
        axes.set_facecolor("white")
        axes.set(xlabel=xlabel, ylabel="Plaque count")
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        if values.size:
            bins = 1 if values.min() == values.max() else min(35, int(np.ceil(np.sqrt(values.size))))
            axes.hist(values, bins=bins, color="#32778B", edgecolor="white", linewidth=0.6)
            mean = np.mean(values)
            median = np.median(values)
            std = np.std(values)
            axes.axvline(
                mean, color="#B64B20", linestyle=(0, (5, 3)), linewidth=2.5,
                label=f"Mean {mean:.2f} ± {std:.2f} (SD)",
            )
            axes.axvline(
                median, color="#272136", linestyle=(0, (2, 2)), linewidth=1.7,
                label=f"Median {median:.2f}",
            )
            axes.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, fontsize=12)
        else:
            axes.set(xticks=[], yticks=[])
            axes.text(0.5, 0.5, empty_message, transform=axes.transAxes, ha="center", va="center")
        figure.savefig(destination, dpi=300)


def export_plaque_statistics(image_path, labels):
    ids, areas, ratios = measure_plaques(labels)
    total_count = len(ids)
    output = image_path.parent / "out"
    output.mkdir(exist_ok=True)
    stem = image_path.stem
    csv_name = f"{stem}_plaque_statistics.csv"
    area_name = f"{stem}_plaque_area_distribution.png"
    ratio_name = f"{stem}_plaque_aspect_ratio_distribution.png"

    # Finish all three files before replacing an earlier export.
    with TemporaryDirectory(prefix=".plaque-statistics-", dir=output) as directory:
        staging = Path(directory)
        with (staging / csv_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("plaque_id", "area_px2", "aspect_ratio"))
            writer.writerows(
                (plaque_id, area, ratio if np.isfinite(ratio) else "")
                for plaque_id, area, ratio in zip(ids, areas, ratios)
            )
        save_histogram(
            staging / area_name, areas, "Plaque area (px²)", "No plaques", total_count=total_count,
        )
        save_histogram(
            staging / ratio_name, ratios[np.isfinite(ratios)],
            "Plaque aspect ratio (major/minor)", "No defined aspect ratios",
            total_count=total_count,
        )
        for name in (csv_name, area_name, ratio_name):
            (staging / name).replace(output / name)
    return total_count
