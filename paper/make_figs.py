#!/usr/bin/env python3
"""Draw both paper figures at ICASSP rules: Times, >= 9 pt text, TrueType (no Type 3), one column wide.

Fig. 1 reads the Lychee-FD per-scenario placement counts (lychee/results/*_full.json).
Fig. 2 uses the Freeze-Omni four-system dose table (n = 1000 per SNR; fo_tables_harsh1000_m*.txt).
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
COL_IN = 3.39  # ICASSP column width
HERE = pathlib.Path(__file__).resolve().parent
RES = pathlib.Path("/Volumes/T7-Data/rspeech/behaviour-aligned-frontends/lychee/results")


def fig_placement():
    rows = [("clean", "L0_full"), ("noise", "N0_full"), ("+RN-t frozen", "Rindep_full"),
            ("+RN-t behav.", "Rjoint_r0_full")]
    classes = [("on_time", "on-time", "#2e7d32"), ("premature", "premature", "#ef6c00"),
               ("mis_addr", "mis-addressed", "#c62828"), ("break_preempt", "break-preempt", "#8e24aa"),
               ("miss", "miss", "#9e9e9e"), ("late", "late", "#607d8b")]
    data = []
    for label, f in rows:
        d = json.loads((RES / f"{f}.json").read_text())
        n = sum(v["n"] for v in d.values())
        tot = {}
        for v in d.values():
            for k, c in v.items():
                if k != "n":
                    tot[k] = tot.get(k, 0) + c
        data.append((label, {k: 100 * tot.get(k, 0) / n for k, _, _ in classes}))
    fig, ax = plt.subplots(figsize=(COL_IN, 2.25))
    for i, (label, pct) in enumerate(data):
        y = len(data) - 1 - i
        left = 0.0
        for k, _, color in classes:
            w = pct[k]
            ax.barh(y, w, left=left, color=color, height=0.62, edgecolor="white", linewidth=0.4)
            if k in ("premature", "miss") and w >= 12:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", color="white", fontweight="bold")
            if k == "on_time" and w >= 5.5:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", color="white", fontweight="bold")
            left += w
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([r[0] for r in reversed(data)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("first-response placement (%)", labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in classes]
    fig.legend(handles, [n for _, n, _ in classes], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 1.0), handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.80), pad=0.2)
    fig.savefig(HERE / "fig_placement.pdf")


def fig_fo_dose():
    snr = [-5, -10, -15, -20]
    sys_ = [("no front-end", [37.7, 32.9, 28.4, 23.1], [46.3, 51.2, 56.5, 63.0], "k", "o"),
            ("frozen", [32.4, 28.8, 24.3, 20.0], [51.9, 55.7, 61.2, 66.6], "tab:blue", "s"),
            ("aligned", [38.2, 36.9, 35.8, 30.7], [46.1, 47.1, 49.0, 54.5], "tab:red", "^"),
            ("ES", [30.8, 27.1, 24.7, 16.3], [53.5, 57.6, 60.8, 70.9], "tab:gray", "v")]
    fig, ax = plt.subplots(1, 2, figsize=(COL_IN, 2.05), sharex=True)
    for name, ot, pr, c, m in sys_:
        ax[0].plot(snr, pr, marker=m, ms=3.2, lw=1.1, color=c, label=name)
        ax[1].plot(snr, ot, marker=m, ms=3.2, lw=1.1, color=c, label=name)
    ax[0].axhline(43.8, color="k", lw=0.6, ls=":")
    ax[1].axhline(40.6, color="k", lw=0.6, ls=":")
    ax[0].set_ylabel("premature (%)", labelpad=1)
    ax[1].set_ylabel("on-time (%)", labelpad=1)
    for a in ax:
        a.set_xlabel("SNR (dB)", labelpad=1)
        a.set_xticks(snr)
        a.invert_xaxis()
        a.grid(alpha=0.3, lw=0.4)
        a.tick_params(pad=1.5)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.4, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.2, w_pad=0.8)
    fig.savefig(HERE / "fig_fo_dose.pdf")


if __name__ == "__main__":
    fig_placement()
    fig_fo_dose()
    print("figures written")
