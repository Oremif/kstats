#!/usr/bin/env python3
"""
Generate SVG charts from kstats JMH benchmark results.

Usage:
    python generate_charts.py                          # auto-discover latest report
    python generate_charts.py path/to/main.json        # explicit path
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

KSTATS_COLOR = "#26A69A"
COMMONS_COLOR = "#FF7043"
FASTER_COLOR = "#43A047"
SLOWER_COLOR = "#E53935"
BAR_HEIGHT = 0.35

SUBTITLE = "OpenJDK 21 · JMH 1.37 · Apple Silicon · 5 warmup x 3 s, 5 measurement x 3 s"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    fqn: str
    class_name: str
    method: str
    package: str
    params: dict
    score: float
    score_error: float
    score_unit: str

    @property
    def is_kstats(self) -> bool:
        return self.method.startswith("kstats")

    @property
    def is_commons(self) -> bool:
        return self.method.startswith("commons")

    @property
    def operation(self) -> str:
        if self.method.startswith("kstats"):
            return self.method[6:]
        if self.method.startswith("commons"):
            return self.method[7:]
        return self.method

    @property
    def param_key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(self.params.items()))


@dataclass
class BenchmarkPair:
    operation: str
    params: dict
    kstats: BenchmarkEntry | None = None
    commons: BenchmarkEntry | None = None

    @property
    def speedup(self) -> float | None:
        if self.kstats and self.commons and self.kstats.score > 0:
            return self.commons.score / self.kstats.score
        return None


def load_data(json_path: str) -> list[BenchmarkEntry]:
    with open(json_path) as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        fqn = item["benchmark"]
        parts = fqn.rsplit(".", 1)
        class_fqn, method = parts[0], parts[1]
        class_name = class_fqn.rsplit(".", 1)[1]
        package = class_fqn.split("benchmark.")[-1].rsplit(".", 1)[0]
        entries.append(BenchmarkEntry(
            fqn=fqn, class_name=class_name, method=method, package=package,
            params=item.get("params", {}),
            score=item["primaryMetric"]["score"],
            score_error=item["primaryMetric"]["scoreError"],
            score_unit=item["primaryMetric"]["scoreUnit"],
        ))
    return entries


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "descriptive.DescriptiveStatsBenchmark": "descriptive",
    "descriptive.QuantileBenchmark": "quantile",
    "correlation.CorrelationBenchmark": "correlation",
    "correlation.CovarianceBenchmark": "correlation",
    "hypothesis.HypothesisTestBenchmark": "hypothesis",
    "hypothesis.PairedTTestBenchmark": "hypothesis",
    "hypothesis.ChiSquaredIndependenceBenchmark": "hypothesis_chi2",
}

DISTRIBUTION_CLASSES = {
    "ContinuousDistributionBenchmark", "DiscreteDistributionBenchmark",
    "FDistributionBenchmark", "CauchyBenchmark", "UniformContinuousBenchmark",
    "WeibullBenchmark", "LogNormalBenchmark", "GeometricBenchmark",
    "HypergeometricBenchmark", "ZipfDistributionBenchmark",
}

CONTINUOUS_DISTS = {"Normal", "Beta", "Gamma", "ChiSq", "StudentT", "Exponential", "F", "Cauchy", "Uniform", "Weibull", "LogNormal"}
DISCRETE_DISTS = {"Binomial", "Poisson", "Geometric", "Hypergeometric", "Zipf"}


def categorize(entries: list[BenchmarkEntry]) -> dict[str, list[BenchmarkEntry]]:
    cats: dict[str, list[BenchmarkEntry]] = defaultdict(list)
    for e in entries:
        key = f"{e.package}.{e.class_name}"
        if key in CATEGORY_MAP:
            cats[CATEGORY_MAP[key]].append(e)
        elif e.class_name in DISTRIBUTION_CLASSES:
            cats["distributions"].append(e)
        else:
            cats["other"].append(e)
    return dict(cats)


def pair_benchmarks(entries: list[BenchmarkEntry]) -> list[BenchmarkPair]:
    groups: dict[str, BenchmarkPair] = {}
    for e in entries:
        key = f"{e.operation}|{e.param_key}"
        if key not in groups:
            groups[key] = BenchmarkPair(operation=e.operation, params=dict(e.params))
        pair = groups[key]
        if e.is_kstats:
            pair.kstats = e
        elif e.is_commons:
            pair.commons = e
    return sorted(groups.values(), key=lambda p: (p.operation, str(sorted(p.params.items()))))


def filter_by_size(pairs: list[BenchmarkPair], size: str, field: str = "size") -> list[BenchmarkPair]:
    return [p for p in pairs if p.params.get(field) == size]


def get_largest_size(pairs: list[BenchmarkPair], field: str = "size") -> str:
    sizes = {p.params.get(field, "0") for p in pairs}
    return max(sizes, key=lambda x: int(x))


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _format_value(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.1f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


LABEL_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)


def _make_bar_chart(
    ax: plt.Axes,
    labels: list[str],
    kstats_vals: list[float],
    commons_vals: list[float],
    kstats_errors: list[float],
    commons_errors: list[float],
    unit: str = "\u03bcs/op",
    log_scale: bool = False,
):
    y = np.arange(len(labels))

    bars_commons = ax.barh(y + BAR_HEIGHT / 2, commons_vals, BAR_HEIGHT,
                           xerr=commons_errors, capsize=2,
                           label="Commons Math", color=COMMONS_COLOR, alpha=0.85,
                           error_kw={"linewidth": 0.8, "alpha": 0.5})
    bars_kstats = ax.barh(y - BAR_HEIGHT / 2, kstats_vals, BAR_HEIGHT,
                          xerr=kstats_errors, capsize=2,
                          label="kstats", color=KSTATS_COLOR, alpha=0.85,
                          error_kw={"linewidth": 0.8, "alpha": 0.5})

    # Compute right margin for label placement
    max_val = max(max(kstats_vals, default=0), max(commons_vals, default=0))
    if log_scale:
        pad = lambda w: w * 0.2
    else:
        pad = lambda w: max_val * 0.02

    # Value labels with white background to avoid grid overlap
    for bar, val in zip(bars_kstats, kstats_vals):
        if val > 0:
            ax.text(bar.get_width() + pad(bar.get_width()),
                    bar.get_y() + bar.get_height() / 2,
                    _format_value(val), va="center", ha="left", fontsize=8,
                    color="#333", bbox=LABEL_BBOX)

    for bar, val in zip(bars_commons, commons_vals):
        if val > 0:
            ax.text(bar.get_width() + pad(bar.get_width()),
                    bar.get_y() + bar.get_height() / 2,
                    _format_value(val), va="center", ha="left", fontsize=8,
                    color="#333", bbox=LABEL_BBOX)

    # Extend x-axis to make room for labels
    if not log_scale and max_val > 0:
        ax.set_xlim(right=max_val * 1.25)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{unit}  (lower is better)", fontsize=10)
    ax.invert_yaxis()

    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Legend outside the chart, top-right corner
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, fontsize=10)


def _finalize_chart(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str = SUBTITLE):
    """Place title and subtitle above axes using fixed point offsets (not figure-relative)."""
    # Subtitle just above the axes, offset 6pt
    sub_trans = ax.transAxes + ScaledTranslation(0, 6 / 72, fig.dpi_scale_trans)
    ax.text(0.5, 1.0, subtitle, transform=sub_trans, ha="center", va="bottom",
            fontsize=8, color="#888")
    # Title above subtitle: 6pt pad + subtitle height + 4pt gap
    n_lines = subtitle.count("\n") + 1
    title_offset = 6 + n_lines * 11 + 4
    title_trans = ax.transAxes + ScaledTranslation(0, title_offset / 72, fig.dpi_scale_trans)
    ax.text(0.5, 1.0, title, transform=title_trans, ha="center", va="bottom",
            fontsize=13, fontweight="bold")


def _save_svg(fig: plt.Figure, path: str):
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=96)
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {os.path.basename(path):40s} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _size_label(size: str, param_field: str) -> str:
    n = int(size)
    if param_field == "tableSize":
        return f"{size}x{size}"
    if n >= 1000:
        return f"n={n // 1000}K"
    return f"n={n}"


def generate_size_charts(
    pairs: list[BenchmarkPair],
    title_prefix: str,
    filename_prefix: str,
    output_dir: str,
    param_field: str = "size",
) -> list[str]:
    """Generate one SVG per parameter value. Returns list of filenames.

    If there are few operations (<=2), merge all sizes into a single chart.
    """
    sizes = sorted({p.params.get(param_field, "0") for p in pairs}, key=lambda x: int(x))

    # Count unique operations
    ops = {p.operation for p in pairs}

    # Merge all sizes into one chart when there are few operations
    if len(ops) <= 2:
        return _generate_merged_chart(pairs, title_prefix, filename_prefix, output_dir, sizes, param_field)

    filenames = []
    for size in sizes:
        group = filter_by_size(pairs, size, param_field)
        if not group:
            continue

        labels = [p.operation for p in group]
        k_vals = [p.kstats.score if p.kstats else 0 for p in group]
        c_vals = [p.commons.score if p.commons else 0 for p in group]
        k_errs = [p.kstats.score_error if p.kstats else 0 for p in group]
        c_errs = [p.commons.score_error if p.commons else 0 for p in group]

        sl = _size_label(size, param_field)

        height = max(3, len(labels) * 0.55 + 1.2)
        fig, ax = plt.subplots(figsize=(10, height))
        _make_bar_chart(ax, labels, k_vals, c_vals, k_errs, c_errs)
        _finalize_chart(fig, ax, f"{title_prefix}  ({sl})")

        fname = f"{filename_prefix}_{size}.svg"
        _save_svg(fig, os.path.join(output_dir, fname))
        filenames.append(fname)

    return filenames


def _generate_merged_chart(
    pairs: list[BenchmarkPair],
    title_prefix: str,
    filename_prefix: str,
    output_dir: str,
    sizes: list[str],
    param_field: str,
) -> list[str]:
    """Merge all sizes into a single chart with labels like 'Operation (n=10K)'."""
    all_items = []
    for size in sizes:
        group = filter_by_size(pairs, size, param_field)
        sl = _size_label(size, param_field)
        for p in group:
            all_items.append((f"{p.operation}  ({sl})", p))

    labels = [it[0] for it in all_items]
    k_vals = [it[1].kstats.score if it[1].kstats else 0 for it in all_items]
    c_vals = [it[1].commons.score if it[1].commons else 0 for it in all_items]
    k_errs = [it[1].kstats.score_error if it[1].kstats else 0 for it in all_items]
    c_errs = [it[1].commons.score_error if it[1].commons else 0 for it in all_items]

    height = max(3, len(labels) * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(10, height))
    _make_bar_chart(ax, labels, k_vals, c_vals, k_errs, c_errs)
    _finalize_chart(fig, ax, title_prefix)

    fname = f"{filename_prefix}.svg"
    _save_svg(fig, os.path.join(output_dir, fname))
    return [fname]


def generate_distribution_charts(entries: list[BenchmarkEntry], output_dir: str) -> list[str]:
    """Generate separate SVGs for continuous and discrete distributions."""
    pairs = pair_benchmarks(entries)
    filenames = []

    # Parse distribution name from operation
    def dist_name(op: str) -> str:
        for suffix in ("Pdf", "Pmf", "Cdf", "Quantile", "QuantileInt"):
            if op.endswith(suffix):
                return op[:-len(suffix)]
        return op

    # Split continuous vs discrete
    continuous = [p for p in pairs if dist_name(p.operation) in CONTINUOUS_DISTS]
    discrete = [p for p in pairs if dist_name(p.operation) in DISCRETE_DISTS]

    for group, group_name in [(continuous, "continuous"), (discrete, "discrete")]:
        if not group:
            continue

        # Sort by distribution name, then operation
        group.sort(key=lambda p: (dist_name(p.operation), p.operation))

        # Make labels like "Normal — pdf" instead of "NormalPdf"
        labels = []
        for p in group:
            dn = dist_name(p.operation)
            op_suffix = p.operation[len(dn):]
            labels.append(f"{dn} — {op_suffix.lower()}")

        k_vals = [p.kstats.score if p.kstats else 0 for p in group]
        c_vals = [p.commons.score if p.commons else 0 for p in group]
        k_errs = [p.kstats.score_error if p.kstats else 0 for p in group]
        c_errs = [p.commons.score_error if p.commons else 0 for p in group]

        # Use log scale if range spans > 2 orders of magnitude
        all_vals = [v for v in k_vals + c_vals if v > 0]
        use_log = (max(all_vals) / min(all_vals)) > 100 if all_vals else False

        height = max(4, len(labels) * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(10, height))

        title = f"{'Continuous' if group_name == 'continuous' else 'Discrete'} Distributions"
        _make_bar_chart(ax, labels, k_vals, c_vals, k_errs, c_errs, log_scale=use_log)
        _finalize_chart(fig, ax, title)

        fname = f"distributions_{group_name}.svg"
        _save_svg(fig, os.path.join(output_dir, fname))
        filenames.append(fname)

    return filenames


def generate_overview_chart(all_entries: list[BenchmarkEntry], output_dir: str) -> str:
    """Diverging bar chart showing speedup ratios, only significant differences (>1.5x)."""
    pairs = pair_benchmarks(all_entries)

    # Keep only the largest size per operation
    best: dict[str, BenchmarkPair] = {}
    for p in pairs:
        op = p.operation
        size = int(p.params.get("size", p.params.get("tableSize", "0")))
        prev = best.get(op)
        if prev is None:
            best[op] = p
        else:
            prev_size = int(prev.params.get("size", prev.params.get("tableSize", "0")))
            if size > prev_size:
                best[op] = p

    # Compute speedup ratios, filter to >1.5x in either direction
    items = []
    for op, p in best.items():
        if p.kstats and p.commons and p.kstats.score > 0 and p.commons.score > 0:
            ratio = p.commons.score / p.kstats.score
            if ratio >= 1.5 or ratio <= (1 / 1.5):
                items.append((op, ratio, p))

    # Sort by speedup (kstats fastest first)
    items.sort(key=lambda x: x[1], reverse=True)

    labels = [it[0] for it in items]
    log_ratios = [math.log2(it[1]) for it in items]
    colors = [FASTER_COLOR if v >= 0 else SLOWER_COLOR for v in log_ratios]

    height = max(5, len(labels) * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))

    y = np.arange(len(labels))
    bars = ax.barh(y, log_ratios, 0.6, color=colors, alpha=0.85)

    # Value labels: show "Nx faster" text
    for bar, ratio_val, log_val in zip(bars, [it[1] for it in items], log_ratios):
        if ratio_val >= 1:
            text = f"{ratio_val:.1f}x" if ratio_val < 100 else f"{ratio_val:.0f}x"
            x_pos = bar.get_width() + 0.15
            ha = "left"
        else:
            text = f"{1/ratio_val:.1f}x"
            x_pos = bar.get_width() - 0.15
            ha = "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                text, va="center", ha=ha, fontsize=8, color="#333",
                bbox=LABEL_BBOX)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("log₂(Commons / kstats)", fontsize=10)
    ax.axvline(0, color="#666", linewidth=0.8)
    ax.invert_yaxis()

    # Annotations at bottom
    ax.text(0.99, 0.02, "kstats faster >>>",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=FASTER_COLOR, fontweight="bold", alpha=0.7)
    ax.text(0.01, 0.02, "<<< Commons faster",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=SLOWER_COLOR, fontweight="bold", alpha=0.7)

    overview_subtitle = "Operations with >1.5x difference at largest tested size\n" + SUBTITLE
    _finalize_chart(fig, ax, "Overall Speedup: kstats vs Commons Math", overview_subtitle)

    fname = "overview.svg"
    _save_svg(fig, os.path.join(output_dir, fname))
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_report() -> str:
    script_dir = Path(__file__).resolve().parent
    benchmark_dir = script_dir.parent
    pattern = str(benchmark_dir / "build" / "reports" / "benchmarks" / "main" / "*" / "main.json")
    reports = sorted(glob.glob(pattern))
    if not reports:
        print("ERROR: No benchmark report found. Run ./gradlew :benchmark:benchmark first.")
        sys.exit(1)
    return reports[-1]


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_report()
    print(f"Reading report: {json_path}")

    output_dir = str(Path(__file__).resolve().parent)
    entries = load_data(json_path)
    print(f"Loaded {len(entries)} benchmark entries\n")

    cats = categorize(entries)
    print("Generating SVG charts:")

    all_files = []

    # Descriptive stats — one chart per size
    if "descriptive" in cats:
        pairs = pair_benchmarks(cats["descriptive"])
        all_files += generate_size_charts(pairs, "Descriptive Statistics", "descriptive", output_dir)

    # Quantile — one chart per size
    if "quantile" in cats:
        pairs = pair_benchmarks(cats["quantile"])
        all_files += generate_size_charts(pairs, "Quantile Benchmarks", "quantile", output_dir)

    # Correlation — one chart per size
    if "correlation" in cats:
        pairs = pair_benchmarks(cats["correlation"])
        all_files += generate_size_charts(pairs, "Correlation & Regression", "correlation", output_dir)

    # Hypothesis tests — one chart per size
    if "hypothesis" in cats:
        pairs = pair_benchmarks(cats["hypothesis"])
        all_files += generate_size_charts(pairs, "Hypothesis Tests", "hypothesis", output_dir)

    # Chi-squared independence — one chart per table size
    if "hypothesis_chi2" in cats:
        pairs = pair_benchmarks(cats["hypothesis_chi2"])
        all_files += generate_size_charts(
            pairs, "Chi-Squared Independence", "chi2_independence", output_dir, "tableSize"
        )

    # Distributions — split continuous/discrete
    if "distributions" in cats:
        all_files += generate_distribution_charts(cats["distributions"], output_dir)

    # Overview
    all_files.append(generate_overview_chart(entries, output_dir))

    print(f"\n{len(all_files)} charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
