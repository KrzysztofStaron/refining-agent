"""Render a brain-optimizer PNG from a results.json file.

Decoupled from inference: this script needs no GPU and never loads TRIBE v2 /
Llama. It only reads the activations that `visualize.py` already computed and
draws the figure, so you can re-style the plot endlessly without re-running the
expensive model.

Usage:
    python render.py [results.json] [out.png]
"""
import sys
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _render_brain_to_axes(signals, axes, views, vmin, vmax):
    from tribev2.plotting.cortical import PlotBrainNilearn
    PlotBrainNilearn().plot_surf(
        signals,
        axes=axes,
        views=views,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        norm_percentile=1,
    )


def render(json_path: str = "results.json", out_path: str = "brain_optimizer.png"):
    with open(json_path) as f:
        data = json.load(f)

    views = data["views"]
    results = data["results"]
    # rebuild activation arrays from the stored lists
    for r in results:
        r["activation"] = np.asarray(r["activation"], dtype=float)

    # consistent color scale across all brains
    all_acts = np.stack([r["activation"] for r in results])
    vmax = float(np.percentile(np.abs(all_acts), 99))
    vmin = 0.0

    n = len(results)
    n_brain_cols = len(views)
    col_ratios = [4] + [1.2] * n_brain_cols + [0.9]
    total_cols = 1 + n_brain_cols + 1

    fig = plt.figure(figsize=(4 + 1.5 * n_brain_cols * n, 3.2 * n), facecolor="#0f0f0f")
    fig.text(
        0.5, 0.98,
        "X Post Brain Activation Optimizer",
        ha="center", va="top", fontsize=30, fontweight="bold",
        color="white", fontfamily="monospace",
    )

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.15, top=0.93, bottom=0.03, left=0.02, right=0.98)

    for row_idx, r in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, total_cols, subplot_spec=outer[row_idx],
            width_ratios=col_ratios, wspace=0.02,
        )

        is_winner = r["rank"] == 1
        border_color = "#f5c518" if is_winner else "#2a2a2a"
        bg_color = "#1a1a1a" if is_winner else "#141414"

        # text panel
        ax_text = fig.add_subplot(inner[0])
        ax_text.set_facecolor(bg_color)
        for spine in ax_text.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2 if is_winner else 0.5)
        ax_text.set_xticks([])
        ax_text.set_yticks([])

        rank_label = f"#{r['rank']} {'WINNER' if is_winner else ''}"
        ax_text.text(0.5, 0.93, rank_label, transform=ax_text.transAxes,
                     ha="center", va="top", fontsize=18, fontweight="bold",
                     color="#f5c518" if is_winner else "#888888", fontfamily="monospace")
        ax_text.text(0.5, 0.80, r["label"], transform=ax_text.transAxes,
                     ha="center", va="top", fontsize=13, color="#aaaaaa", fontfamily="monospace")

        post_text = r["post"]
        if len(post_text) > 200:
            post_text = post_text[:199].rstrip() + "…"
        wrapped = "\n".join(textwrap.wrap(post_text, width=34))
        ax_text.text(0.5, 0.65, wrapped, transform=ax_text.transAxes,
                     ha="center", va="top", fontsize=16,
                     color="white", wrap=True, linespacing=1.4)

        # brain views
        brain_axes = []
        for col_i in range(n_brain_cols):
            ax_b = fig.add_subplot(inner[1 + col_i], projection="3d")
            ax_b.set_facecolor("#0f0f0f")
            brain_axes.append(ax_b)

        _render_brain_to_axes(r["activation"], brain_axes, views, vmin, vmax)

        if is_winner:
            for ax_b in brain_axes:
                for spine in ax_b.spines.values():
                    spine.set_edgecolor("#f5c518")

        # score panel
        ax_score = fig.add_subplot(inner[-1])
        ax_score.set_facecolor(bg_color)
        for spine in ax_score.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2 if is_winner else 0.5)
        ax_score.set_xticks([])
        ax_score.set_yticks([])
        ax_score.text(0.5, 0.6, f"{r['score']:.4f}", transform=ax_score.transAxes,
                      ha="center", va="center", fontsize=20, fontweight="bold",
                      color="#f5c518" if is_winner else "#dddddd", fontfamily="monospace")
        ax_score.text(0.5, 0.35, "brain\nscore", transform=ax_score.transAxes,
                      ha="center", va="center", fontsize=11, color="#666666", fontfamily="monospace")

    # colorbar
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar_ax = fig.add_axes([0.985, 0.1, 0.008, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(colors="white", labelsize=11)
    cbar.set_label("activation", color="white", fontsize=14)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")

    winner = min(results, key=lambda x: x["rank"])
    print(f"\n🏆 Winner: {winner['label']} (score={winner['score']:.4f})")
    print(f"   {winner['post']}")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "brain_optimizer.png"
    render(json_path, out_path)
