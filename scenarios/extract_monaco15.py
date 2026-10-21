import argparse
import os
import subprocess
import sys
from pathlib import Path


def sumo_tool(name):
    home = os.environ.get("SUMO_HOME")
    if not home:
        raise SystemExit("SUMO_HOME is not set")
    return Path(home) / "tools" / name


def derive_edges(net_file, tls_file, out_file):
    """Edge list of the corridor: every edge entering or leaving one of the listed junctions, plus the
    shortest path (by length) between consecutive junctions of the list, which is given in driving order."""
    import sumolib
    net = sumolib.net.readNet(net_file, withInternal=False)
    ids = [l.split("#")[0].strip() for l in Path(tls_file).read_text().splitlines() if l.split("#")[0].strip()]
    nodes = [net.getNode(i) for i in ids]
    keep = set()
    for n in nodes:
        keep.update(e.getID() for e in n.getIncoming() + n.getOutgoing())
    for a, b in zip(nodes[:-1], nodes[1:]):
        path, _ = net.getShortestPath(a.getOutgoing()[0], b.getIncoming()[0])
        keep.update(e.getID() for e in (path or []))
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text("\n".join(sorted(keep)) + "\n")
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--most-net", required=True)
    ap.add_argument("--most-routes", nargs="+", required=True)
    ap.add_argument("--edges", default="data/monaco15/edges.txt")
    ap.add_argument("--tls-file", default=None, help="15 junction ids, one per line, west to east; derives --edges")
    ap.add_argument("--out", default="data/monaco15")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.tls_file:
        derive_edges(a.most_net, a.tls_file, a.edges)
    edges = [l.strip() for l in Path(a.edges).read_text().split() if l.strip()]
    net = out / "monaco15.net.xml"
    subprocess.check_call(["netconvert", "-s", a.most_net, "--keep-edges.input-file", a.edges,
                           "--tls.guess-signals", "false", "-o", str(net)])
    routes = out / "monaco15.rou.xml"
    subprocess.check_call([sys.executable, str(sumo_tool("route/cutRoutes.py")), str(net), *a.most_routes,
                           "--routes-output", str(routes), "--orig-net", a.most_net])
    cfg = out / "monaco15.sumocfg"
    cfg.write_text(
        '<configuration>\n  <input>\n    <net-file value="monaco15.net.xml"/>\n'
        '    <route-files value="monaco15.rou.xml"/>\n  </input>\n'
        '  <time>\n    <begin value="0"/>\n    <end value="3600"/>\n  </time>\n</configuration>\n')
    import sumolib
    n_tls = len(sumolib.net.readNet(str(net)).getTrafficLights())
    print(f"{len(edges)} edges, {n_tls} signals -> {net}, {routes}, {cfg}")
    if n_tls != 15:
        print(f"warning: expected 15 signals, found {n_tls}", file=sys.stderr)


if __name__ == "__main__":
    main()
