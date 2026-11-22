"""Stand-in for cp_verify that runs two builds on the same arguments and compares them.

scripts/flashprove_tinyllama.py copies this file into a run folder with its own interpreter as the
first line, and hands that copy to upstream's scripts in place of cp_verify. Each call runs the
unpatched build (DUAL_UPSTREAM) and the patched build (DUAL_PATCHED), appends
one JSON line to DUAL_LOG, and passes on the patched build's output and exit code. If the two differ
in exit code, standard output or standard error, beyond the build digest and wall-clock times, it
exits with 99 instead, so that no caller can mistake the call for an ordinary result.
"""

import json
import os
import re
import subprocess
import sys

# Every printf format of cp_verify that prints a wall-clock time (%.3f), copied from the pinned
# upstream sources (verifier/src: cp_verify.cpp, vf_segments.inc, vf_layers_check.inc,
# vf_output_head_check.inc). A line that matches one of them in full has only its times replaced;
# every other line, and every other field of these lines, must be equal byte for byte.
# scripts/flashprove_tinyllama.py checks that the sources hold exactly these 14 times.
TIMED_FORMATS = (
    "---- stage %s ---- %d thread(s), %.3f s",
    "ok public inputs: parameter file, shape, no-overflow conditions; profile %u; %llu blocks of"
    " layers  %.3f s",
    "read: sub-proof A_%llu: %u claims listed (tested at the chain's end), %u slots  %.3f s",
    "read: sub-proof G_%llu: %zu lookup calls in one forest, 2 more in the smallness test's, %zu"
    " matrices, %u claims listed (tested at the chain's end), %u slots  %.3f s",
    "read: sub-proof H: %zu lookup calls in one forest, 2 more in the smallness test's, %zu"
    " matrices, %u claims listed (tested at the chain's end), %u slots  %.3f s",
    "   seconds: the frame's first() %.3f | the sub-proof's own checks %.3f | the frame's later()"
    " %.3f | the equation's terms made %.3f | the frame's close() %.3f",
    "seconds: the links' last proximity test (the frame's finish()) %.3f",
    "seconds: before the proof is read (the scan, the plans, blk::configure(), the request's form,"
    " the accounting) %.3f",
    "seconds: the links with their last test %.3f | behind them (the rows on the table, the place,"
    " the shared chain, the second chain) %.3f",
)
CONVERSIONS = {
    "%.3f": rb"(\d+\.\d{3})",
    "%llu": rb"\d+",
    "%zu": rb"\d+",
    "%u": rb"\d+",
    "%d": rb"\d+",
}
CONVERSIONS["%s"] = rb"[a-z]+"


def pattern(fmt):
    """A printf format as a regular expression for one whole line; each %.3f is a group."""
    pieces = re.split(r"(%\.3f|%llu|%zu|%u|%d|%s)", fmt)
    body = b"".join(CONVERSIONS.get(p) or re.escape(p.encode()) for p in pieces)
    return re.compile(b"^" + body + b"$")


TIMED = [pattern(f) for f in TIMED_FORMATS]


def normal_line(line):
    for timed in TIMED:
        match = timed.match(line)
        if match:
            out, at = b"", 0
            for g in range(1, timed.groups + 1):
                out += line[at : match.start(g)] + b"<t>"
                at = match.end(g)
            return out + line[at:]
    return line


def normal(output, digest):
    """The output with the build digest and the times of cp_verify's timed lines replaced."""
    lines = output.replace(digest.encode(), b"<BUILD>").split(b"\n")
    return b"\n".join(normal_line(x) for x in lines)


def run(binary):
    done = subprocess.run([binary, *sys.argv[1:]], capture_output=True, timeout=1800)
    return done.returncode, done.stdout, done.stderr


def main():
    upstream, patched = os.environ["DUAL_UPSTREAM"], os.environ["DUAL_PATCHED"]
    digests = os.environ["DUAL_UPSTREAM_DIGEST"], os.environ["DUAL_PATCHED_DIGEST"]
    (code_u, out_u, err_u), (code_d, out_d, err_d) = run(upstream), run(patched)
    same = (
        code_u == code_d
        and normal(out_u, digests[0]) == normal(out_d, digests[1])
        and normal(err_u, digests[0]) == normal(err_d, digests[1])
    )
    with open(os.environ["DUAL_LOG"], "a", encoding="utf-8") as log:
        log.write(
            json.dumps(dict(arguments=sys.argv[1:], exit_codes=[code_u, code_d], same=same)) + "\n"
        )
    sys.stdout.buffer.write(out_d)
    sys.stderr.buffer.write(err_d)
    sys.exit(code_d if same else 99)


if __name__ == "__main__":
    main()
