"""Compare cp_commit's worst-case table from a GPU with worst_case_cpu.py's result.

usage: python3 -I -B compare_worst_case.py TABLE PORT_JSON OUT_JSON

TABLE is the file that cp_commit writes with -worst-case-table (gpu_job.sh keeps it); PORT_JSON is
worst-case-cpu.json of a run of make verify-flashprove-tinyllama-answer on the same kit. Both list
the same sites (a site's name and layer), each with the parameter that decides it, its value, a
bound and a limit; the table prints log2 of both with three decimals.

Required: the same sites, parameters and values; the GPU's verdict "fits" with no site short; and
at every site that worst_case_cpu.py computes as the GPU does (all but the stream's), bounds and
limits equal within the table's resolution. At the stream's sites the port's bound, by Frobenius
norms, must not be below the GPU's. Exit status 0 if all hold, 1 otherwise; OUT_JSON gets the
figures either way.
"""

import json
import math
import sys
from pathlib import Path

RESOLUTION = 0.0011  # three printed decimals, and their rounding
STREAM = ("stream at norm 1", "stream at norm 2", "stream at the last norm", "last stream, a cell")


def table_sites(text):
    """(site, layer) -> (parameter, value, log2 bound, log2 limit, whole bits short)."""
    sites = {}
    for line in text.splitlines():
        cells = line.split(" | ")
        if cells[0] == "site":
            _, site, layer, word, value, bound, limit, short = cells
            sites[site, int(layer)] = (word, value, float(bound), float(limit), int(short))
    return sites


def compare(table, port):
    gpu = table_sites(table)
    cpu = {(row["site"], row["layer"]): row for row in port["lines"]}
    missed = []
    if set(gpu) != set(cpu):
        return dict(sites=len(gpu)), ["the two list other sites"]
    if "verdict fits" not in table.splitlines():
        missed.append("the GPU's verdict is not 'fits'")
    kinds = {}
    for key, (word, value, bound, limit, short) in gpu.items():
        row = cpu[key]
        if (row["word"], str(row["value"])) != (word, value):
            missed.append(f"{key}: another parameter or value")
        if short or row["shortfall"]:
            missed.append(f"{key}: short")
        above = math.log2(row["bound"]) - bound
        if abs(math.log2(row["limit"]) - limit) > RESOLUTION:
            missed.append(f"{key}: the limits differ")
        if key[0] in STREAM:
            if above < -RESOLUTION:
                missed.append(f"{key}: the port's bound is below the GPU's")
        elif abs(above) > RESOLUTION:
            missed.append(f"{key}: the bounds differ by {above:.4f} bits")
        kind = kinds.setdefault(
            key[0], dict(sites=0, port_above_gpu_bits=[above, above], least_gpu_headroom_bits=9e9)
        )
        kind["sites"] += 1
        kind["port_above_gpu_bits"] = [
            round(min(kind["port_above_gpu_bits"][0], above), 4),
            round(max(kind["port_above_gpu_bits"][1], above), 4),
        ]
        kind["least_gpu_headroom_bits"] = round(
            min(kind["least_gpu_headroom_bits"], limit - bound), 3
        )
    tightest = sorted((round(v[3] - v[2], 3), k[0], k[1]) for k, v in gpu.items())[:4]
    return dict(sites=len(gpu), by_site=kinds, tightest_gpu_sites=tightest), missed


if __name__ == "__main__":
    table, port, out = (Path(x) for x in sys.argv[1:4])
    found, missed = compare(table.read_text(), json.loads(port.read_text()))
    found["missed"] = missed
    out.write_text(json.dumps(found, indent=2) + "\n")
    print("agree" if not missed else "DIFFER: " + "; ".join(missed[:5]))
    sys.exit(1 if missed else 0)
