"""Figures for the README from reports/dual3090/summary.json and metrics.json.

    python scripts/plot_study.py reports/dual3090 docs/figures
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {'keller': '#5b8def', 'gamma_0.20': '#e8883a', 'exact_svd': '#2fae94'}
NAMES = {'keller': 'Keller', 'gamma_0.20': 'γ = 0.20 fit', 'exact_svd': 'Exact SVD'}
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def paired(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, task in zip(axes, ('cifar', 'language')):
        rows = [r for r in metrics[task] if r['arm'] != 'keller']
        for i, row in enumerate(rows):
            lo, hi = row['paired_vs_keller']['ci95']
            mean = row['paired_vs_keller']['mean']
            color = COLORS[row['arm']]
            ax.plot([lo, hi], [i, i], color=color, linewidth=2, solid_capstyle='round')
            ax.plot(mean, i, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
        ax.set_yticks(range(len(rows)), [f"{NAMES[r['arm']]} {r['precision']}" for r in rows])
        ax.set_title('CIFAR-10 CNN' if task == 'cifar' else '124M GPT, FineWeb', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('test loss minus Keller (same seed, same precision)', color=MUTED, fontsize=9)
        style(ax)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, 'Dots: mean of 3 paired seeds. Lines: 95% t-interval. Left of the dashed line beats Keller.',
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out/'paired_vs_keller.png', dpi=180)


def spectra(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, task in zip(axes, ('cifar', 'language')):
        rows = {}
        for r in summary['confirmation']:
            if r['task'] == task and r['precision'] == 'fp32' and r['spectra']:
                rows.setdefault(r['arm'], []).append(r['spectra'])
        for i, arm in enumerate(('exact_svd', 'gamma_0.20', 'keller')):
            q = {k: sum(s[k] for s in rows[arm])/len(rows[arm]) for k in ('p0', 'p25', 'p50', 'p75', 'p100')}
            color = COLORS[arm]
            ax.plot([q['p0'], q['p100']], [i, i], color=color, linewidth=1.5, alpha=0.6)
            ax.plot([q['p25'], q['p75']], [i, i], color=color, linewidth=8, solid_capstyle='butt')
            if q['p75'] - q['p25'] < 0.01:  # exact SVD: every kept direction lands on 1
                ax.plot(q['p50'], i, 'D', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
                ax.annotate('kept directions at 1.0', (q['p50'], i), xytext=(10, 6), textcoords='offset points',
                            color=MUTED, fontsize=8)
            else:
                ax.plot(q['p50'], i, '|', color='white', markersize=10, markeredgewidth=2)
        ax.set_yticks(range(3), [NAMES[a] for a in ('exact_svd', 'gamma_0.20', 'keller')])
        ax.set_title('CIFAR-10 CNN' if task == 'cifar' else '124M GPT, FineWeb', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('singular value of the applied update (fp32 arms)', color=MUTED, fontsize=9)
        ax.set_xlim(0, 1.35)
        style(ax)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, 'Thick: middle 50% of singular values. Thin: min to max. Median over sampled steps and matrices.',
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out/'update_spectra.png', dpi=180)


def tuning(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, task in zip(axes, ('cifar', 'language')):
        for arm in ('keller', 'gamma_0.20', 'exact_svd'):
            cells = sorted((c for c in summary['tuning'] if c['task'] == task and c['arm'] == arm
                            and c['precision'] == 'fp32'), key=lambda c: c['lr'])
            xs = [c['lr'] for c in cells]
            ys = [c['mean_validation_loss'] for c in cells]
            ax.plot(xs, ys, '-o', color=COLORS[arm], linewidth=2, markersize=6, markeredgecolor='white',
                    markeredgewidth=1.5, label=NAMES[arm])
        ax.set_xscale('log')
        ax.set_xticks([0.005, 0.02, 0.08], ['0.005', '0.02', '0.08'])
        ax.minorticks_off()
        ax.set_title('CIFAR-10: best LR is the largest tried' if task == 'cifar'
                     else 'GPT: best LR is the smallest tried', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('Muon learning rate', color=MUTED, fontsize=9)
        ax.set_ylabel('mean validation loss', color=MUTED, fontsize=9)
        style(ax)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out/'lr_tuning.png', dpi=180)


def main(reports, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((reports/'summary.json').read_text())
    metrics = json.loads((reports/'metrics.json').read_text())
    paired(metrics, out)
    spectra(summary, out)
    tuning(summary, out)
    print(f'wrote {out}/paired_vs_keller.png, update_spectra.png, lr_tuning.png')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
