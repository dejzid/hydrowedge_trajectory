"""Run trajectory cases, the equations of motion of the paper, for axial and transverse dipoles.

A case is a JSON file in cases/ (see cases/README.md for the fields): either a set of
trajectories ("kind": "paths", the default) or a state diagram over two parameters
("kind": "scan"). Its results go to results/cases/<name>/ and are never overwritten:
  - same name, same parameters  -> skipped, the existing results are kept;
  - same name, other parameters -> error; give the case a new name.
Velocity profiles are cached in results/profiles/, one file per (dipole, alpha, grid);
an existing profile is reused, never recomputed.

Usage:
    python run_trajectories.py cases/transverse_pi3_bulk.json [more cases ...]
    python run_trajectories.py --all          # every cases/*.json
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import mpmath as mp
import numpy as np
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from hydrowedge.trajectory import DIPOLES, GAP_MIN, N_GRID, Profile, trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
RESULTS = ROOT / "results" / "cases"
PROFILES = ROOT / "results" / "profiles"

DEFAULTS = dict(v0=1.0, d_min=1.0, rho_min=1.0, rho_max=100.0, t_max=1e4, checkpoints=[], arrows=8)
END_MARKER = {"wall": "x", "apex": "s", "far": ">", "t_max": "|"}

# State-diagram outcomes (colour-blind safe colours).
OUTCOMES = ["escapes", "same wall", "opposite wall", "apex", "t_max"]
OUTCOME_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#BBBBBB"]
ANGLE_PARAMS = {"tilt", "heading", "delta0", "beta0"}


def angle(value):
    """A number, or an expression in pi such as "pi/3", evaluated at 30 digits."""
    return mp.mpf(str(sp.N(sp.sympify(str(value)), 30)))


def slug(text):
    return str(text).replace("*", "").replace("/", "_").replace(" ", "")


def profile_for(dipole, alpha_label):
    """Cached profile; computed (~1 min) if the file does not exist yet."""
    gap = GAP_MIN[dipole]
    path = PROFILES / f"{dipole}_alpha={slug(alpha_label)}_n{N_GRID}_gap{gap}.csv"
    if not path.exists():
        print(f"  computing profile {path.name} ...", flush=True)
        PROFILES.mkdir(parents=True, exist_ok=True)
        with mp.workdps(30):
            Profile.compute(dipole, angle(alpha_label), str(alpha_label)).save(path)
    return Profile.load(path), path


def git_version():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    dirty = run("status", "--porcelain", "--", "hydrowedge", "run_trajectories.py")
    return run("rev-parse", "--short", "HEAD") + ("+modified" if dirty else "")


def load_case(path):
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    case.setdefault("name", Path(path).stem)
    kind = case.get("kind", "paths")
    if kind not in ("paths", "scan"):
        raise ValueError(f"{path}: 'kind' must be 'paths' or 'scan'")
    if case.get("dipole") not in DIPOLES:
        raise ValueError(f"{path}: 'dipole' must be one of {DIPOLES}")
    required = ("alpha", "sigma", "starts") if kind == "paths" else ("alpha", "start", "x", "y")
    for key in required:
        if key not in case:
            raise ValueError(f"{path}: missing '{key}'")
    return case


def resolve_start(s, alpha):
    """(rho0, beta0, delta0) as floats.

    Position: rho0 with beta0, or rho0 with d0 = distance to a wall ("wall": "upper"
    (default) or "lower"). Orientation: delta0 (from e_rho), heading (from the x-axis,
    = beta + delta), or along ("outward" / "apexward", parallel to that wall) with tilt
    (rotation towards the wall, positive into it).
    """
    rho0 = angle(s["rho0"])
    if "d0" in s:
        side = -1 if s.get("wall") == "lower" else 1
        beta0 = side * (alpha - mp.asin(angle(s["d0"]) / rho0))
    else:
        beta0 = angle(s["beta0"])
        side = -1 if s.get("wall", "upper" if beta0 >= 0 else "lower") == "lower" else 1
    if "heading" in s:
        delta0 = angle(s["heading"]) - beta0
    elif "along" in s:
        tilt = angle(s.get("tilt", 0))
        if s["along"] == "outward":
            heading = side * (alpha + tilt)
        elif s["along"] == "apexward":
            heading = side * (alpha + mp.pi - tilt)
        else:
            raise ValueError("'along' must be 'outward' or 'apexward'")
        delta0 = heading - beta0
    else:
        delta0 = angle(s.get("delta0", 0))
    return float(rho0), float(beta0), float(delta0)


def repulsion_note(p):
    r = p.get("repulsion")
    return "" if r is None else f", repulsion U = {r['strength']:g}, d_r = {r['range']:g}, n = {r['power']:g}"


def integrate(profile, p, start, sigma):
    rho0, beta0, delta0 = start
    return trajectory(profile, rho0, beta0, delta0, sigma=float(sigma), v0=float(p["v0"]),
                      t_max=p["t_max"], d_min=p["d_min"], rho_min=p["rho_min"], rho_max=p["rho_max"],
                      repulsion=p.get("repulsion"))


# --- kind "paths" -------------------------------------------------------------------

def run_paths(p, profile):
    sigmas = p["sigma"] if isinstance(p["sigma"], list) else [p["sigma"]]
    with mp.workdps(30):
        alpha = angle(p["alpha"])
        p["resolved_starts"] = [resolve_start(s, alpha) for s in p["starts"]]
    return [(float(sigma), i, integrate(profile, p, start, sigma))
            for sigma in sigmas for i, start in enumerate(p["resolved_starts"])]


def _write_csv(path, header, data):
    with open(path, "w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows([[f"{v:.12g}" if isinstance(v, float) else v for v in r] for r in data])


def path_samples(tr, n=500):
    """(t, state) for trajectories.csv: n times uniform in t plus n uniform in path length
    in the cross-section, so that a fast approach (e.g. to a wall, before a slow slide
    along it) is resolved as well as a slow drift."""
    tt = np.union1d(tr.t, tr._sol.t)
    rho, beta = tr._sol.sol(tt)[:2]
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(rho * np.cos(beta)), np.diff(rho * np.sin(beta))))])
    t = np.linspace(0, tr.t[-1], n)
    if s[-1] > 0:
        t = np.union1d(t, np.interp(np.linspace(0, s[-1], n), s, tt))
    return t, tr.state(t)


def write_paths(out, p, runs):
    cols = ["sigma", "start", "t", "rho", "beta", "delta", "z", "x", "y", "wall_distance"]

    def rows(sigma, i, tr, t, state):
        rho, beta, delta, z = state.T
        d = rho * np.sin(tr.alpha - np.abs(beta))
        return [[sigma, i, *vals] for vals in zip(t, rho, beta, delta, z,
                                                  rho * np.cos(beta), rho * np.sin(beta), d)]

    summary, checkpoints, paths = [], [], []
    for sigma, i, tr in runs:
        rho0, beta0, delta0 = p["resolved_starts"][i]
        if p["dipole"] == "axial":
            delta0 = float("nan")
        end = tr.state(tr.t[-1])[0]
        summary.append([sigma, i, rho0, beta0, delta0, tr.reason,
                        float(tr.t[-1]), *map(float, end),
                        float(end[0] * np.sin(tr.alpha - abs(end[1])))])
        times = [t for t in p["checkpoints"] if t <= tr.t[-1]] + [float(tr.t[-1])]
        checkpoints += rows(sigma, i, tr, np.array(times, float), tr.state(times))
        paths += rows(sigma, i, tr, *path_samples(tr))
    _write_csv(out / "summary.csv", ["sigma", "start", "rho0", "beta0", "delta0", "end", "t_end",
                                     "rho_end", "beta_end", "delta_end", "z_end",
                                     "wall_distance_end"], summary)
    _write_csv(out / "checkpoints.csv", cols, checkpoints)
    _write_csv(out / "trajectories.csv", cols, paths)

    # The same checkpoints as a table to paste into a message.
    var = ["t", "rho", "beta", "z"] if p["dipole"] == "axial" else ["t", "rho", "beta", "delta"]
    idx = {c: k for k, c in enumerate(cols)}
    md = [f"# {p['name']}: checkpoints",
          "",
          f"{p['dipole']} dipole, alpha = {p['alpha']}, v0 = {p['v0']:g}, d_min = {p['d_min']:g}"
          f"{repulsion_note(p)}; angles in rad; the last row of each run is where it stopped.",
          ""]
    for sigma, i, tr in runs:
        rho0, beta0, delta0 = p["resolved_starts"][i]
        start = f"rho0 = {rho0:.10g}, beta0 = {beta0:.10g}"
        if p["dipole"] == "transverse":
            start += f", delta0 = {delta0:.10g}"
        md += [f"sigma = {sigma:g}, start {i}: {start} (ends: {tr.reason})", "",
               "| " + " | ".join(var) + " |", "|" + "---|" * len(var)]
        md += ["| " + " | ".join(f"{r[idx[v]]:.10g}" for v in var) + " |"
               for r in checkpoints if r[0] == sigma and r[1] == i]
        md.append("")
    (out / "checkpoints.md").write_text("\n".join(md), encoding="utf-8")


def plot_paths(out, p, profile, runs):
    sigmas = sorted({s for s, _, _ in runs}, reverse=True)
    a = profile.alpha
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.5 * len(sigmas), 5.5),
                             constrained_layout=True, squeeze=False)
    trs = [tr for _, _, tr in runs]
    rho0_max = max(s[0] for s in p["resolved_starts"])
    r_max = p.get("view") or min(1.1 * max(tr.rho.max() for tr in trs), 2.5 * rho0_max)
    for ax, sigma in zip(axes[0], sigmas):
        ax.plot([r_max * np.cos(a), 0, r_max * np.cos(a)],
                [r_max * np.sin(a), 0, -r_max * np.sin(a)], "k", lw=2)
        for s_, i, tr in runs:
            if s_ != sigma:
                continue
            (line,) = ax.plot(tr.x, tr.y, lw=1.2)
            c = line.get_color()
            ax.plot(tr.x[0], tr.y[0], "o", color=c, ms=4)
            ax.plot(tr.x[-1], tr.y[-1], END_MARKER[tr.reason], color=c, ms=7)
            if p["dipole"] == "transverse" and p["arrows"]:
                k = np.linspace(0, len(tr.t) - 1, p["arrows"]).astype(int)
                ax.quiver(tr.x[k], tr.y[k], np.cos(tr.heading[k]), np.sin(tr.heading[k]),
                          color=c, angles="xy", scale=28, width=0.004, headwidth=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.03 * r_max, 1.03 * r_max)
        ax.set_ylim(-1.03 * r_max * np.sin(a), 1.03 * r_max * np.sin(a))
        kind = "pusher" if sigma > 0 else "puller" if sigma < 0 else "no dipole"
        ax.set_title(f"{kind}, σ = {sigma:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        f"{p['name']}: {p['dipole']} dipole, α = {p['alpha']}, v0 = {p['v0']:g}, d_min = {p['d_min']:g}"
        f"{repulsion_note(p)}\n"
        "o start   x wall contact   ■ apex   > ρ_max   | t_max"
        + ("   arrows: swimming direction" if p["dipole"] == "transverse" else "")
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"paths.{ext}", dpi=150)
    plt.close(fig)


# --- kind "scan" (state diagram) ----------------------------------------------------

def axis_values(spec):
    with mp.workdps(30):
        lo, hi = float(angle(spec["from"])), float(angle(spec["to"]))
    return np.linspace(lo, hi, int(spec["n"]))


def outcome(tr, start_beta):
    if tr.reason == "far":
        return "escapes"
    if tr.reason == "wall":
        same = np.sign(tr.beta[-1]) == np.sign(start_beta) or start_beta == 0
        return "same wall" if same else "opposite wall"
    return tr.reason  # "apex" or "t_max"


def run_scan(p, profile):
    xs, ys = axis_values(p["x"]), axis_values(p["y"])
    with mp.workdps(30):
        alpha = angle(p["alpha"])
    rows = []
    for yv in ys:
        for xv in xs:
            q, start = dict(p), dict(p["start"])
            for spec, v in ((p["x"], xv), (p["y"], yv)):
                if spec["param"] in ("sigma", "v0"):
                    q[spec["param"]] = v
                else:
                    start[spec["param"]] = v
            with mp.workdps(30):
                s = resolve_start(start, alpha)
            tr = integrate(profile, q, s, q["sigma"])
            end = tr.state(tr.t[-1])[0]
            rows.append([float(xv), float(yv), outcome(tr, s[1]), tr.reason, float(tr.t[-1]),
                         *map(float, end)])
    return xs, ys, rows


def write_scan(out, p, scan):
    xs, ys, rows = scan
    _write_csv(out / "scan.csv", [p["x"]["param"], p["y"]["param"], "outcome", "end", "t_end",
                                  "rho_end", "beta_end", "delta_end", "z_end"], rows)


def plot_scan(out, p, scan):
    xs, ys, rows = scan
    grid = np.array([OUTCOMES.index(r[2]) for r in rows]).reshape(len(ys), len(xs))
    conv = lambda name, v: np.degrees(v) if name in ANGLE_PARAMS else v  # noqa: E731
    label = lambda name: f"{name} (deg)" if name in ANGLE_PARAMS else name  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.pcolormesh(conv(p["x"]["param"], xs), conv(p["y"]["param"], ys), grid,
                  cmap=ListedColormap(OUTCOME_COLORS), vmin=-0.5, vmax=len(OUTCOMES) - 0.5,
                  shading="nearest")
    present = sorted(set(grid.ravel()))
    ax.legend(handles=[Patch(color=OUTCOME_COLORS[k], label=OUTCOMES[k]) for k in present],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_xlabel(label(p["x"]["param"]))
    ax.set_ylabel(label(p["y"]["param"]))
    s = p["start"]
    ax.set_title(f"{p['name']}: {p['dipole']}, α = {p['alpha']}\nstart {json.dumps(s)}", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"diagram.{ext}", dpi=150)
    plt.close(fig)


# --- driver -------------------------------------------------------------------------

def process(path):
    case = load_case(path)
    out = RESULTS / case["name"]
    if (out / "case.json").exists():
        stored = json.loads((out / "case.json").read_text(encoding="utf-8"))["input"]
        if stored == case:
            print(f"{case['name']}: results exist with the same parameters, kept ({out})")
            return
        raise SystemExit(f"{case['name']}: {out} holds results for different parameters; "
                         "give the case a new name (results are never overwritten)")
    print(f"{case['name']}: running", flush=True)
    p = {**DEFAULTS, **case}
    profile, profile_path = profile_for(p["dipole"], p["alpha"])
    tmp = RESULTS / f".{case['name']}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if p.get("kind", "paths") == "paths":
        runs = run_paths(p, profile)
        write_paths(tmp, p, runs)
        plot_paths(tmp, p, profile, runs)
        ends = [tr.reason for _, _, tr in runs]
    else:
        scan = run_scan(p, profile)
        write_scan(tmp, p, scan)
        plot_scan(tmp, p, scan)
        ends = [r[2] for r in scan[2]]
    meta = {
        "input": case,
        "profile": (profile_path.relative_to(ROOT) if profile_path.is_relative_to(ROOT)
                    else profile_path).as_posix(),
        "code_version": git_version(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (tmp / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.rename(out)  # appears complete or not at all
    counts = {e: ends.count(e) for e in dict.fromkeys(ends)}
    print(f"{case['name']}: done, ends {counts} -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case JSON files")
    ap.add_argument("--all", action="store_true", help="run every cases/*.json")
    args = ap.parse_args(argv)
    paths = sorted(CASES.glob("*.json")) if args.all else [Path(c) for c in args.cases]
    if not paths:
        ap.error("no cases given")
    for path in paths:
        process(path)


if __name__ == "__main__":
    sys.exit(main())
