import argparse
import subprocess
from pathlib import Path

import numpy as np


def poisson_routes(rows, cols, peak_vph, episode_s, seed, path):
    rng = np.random.default_rng(seed)
    entries = [f"top{c}_in" for c in range(cols)] + [f"bot{c}_in" for c in range(cols)] + \
              [f"left{r}_in" for r in range(rows)] + [f"right{r}_in" for r in range(rows)]
    exits = [e.replace("_in", "_out") for e in entries]
    lines = ["<routes>"]
    t = np.arange(0, episode_s)
    profile = 0.5 + 0.5 * np.exp(-((t - episode_s / 2) ** 2) / (2 * (episode_s / 5) ** 2))
    rate = peak_vph / 3600.0 * profile / len(entries)
    n = 0
    veh = []
    for e in entries:
        k = rng.poisson(rate)
        for s in np.flatnonzero(k):
            for _ in range(int(k[s])):
                dst = exits[int(rng.integers(len(exits)))]
                veh.append((float(s), e, dst))
    for s, e, d in sorted(veh):
        lines.append(f'  <trip id="v{n}" depart="{s:.0f}" from="{e}" to="{d}"/>')
        n += 1
    lines.append("</routes>")
    Path(path).write_text("\n".join(lines))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--peak", type=int, default=4800)
    ap.add_argument("--out", default="data/grid4x4")
    ap.add_argument("--lane-length", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = out / f"grid{a.rows}x{a.cols}.net.xml"
    subprocess.check_call(["netgenerate", "--grid", "--grid.x-number", str(a.cols), "--grid.y-number", str(a.rows),
                           "--grid.length", str(a.lane_length), "--grid.attach-length", str(a.lane_length),
                           "--default.lanenumber", "2", "--tls.guess", "true", "--tls.default-type", "static",
                           "-o", str(net)])
    rou = out / f"grid{a.rows}x{a.cols}.rou.xml"
    n = poisson_routes(a.rows, a.cols, a.peak, 3600, a.seed, rou)
    (out / f"grid{a.rows}x{a.cols}.sumocfg").write_text(
        f'<configuration>\n  <input>\n    <net-file value="{net.name}"/>\n    <route-files value="{rou.name}"/>\n'
        '  </input>\n  <time>\n    <begin value="0"/>\n    <end value="3600"/>\n  </time>\n</configuration>\n')
    print(f"{n} trips -> {out}")


if __name__ == "__main__":
    main()
