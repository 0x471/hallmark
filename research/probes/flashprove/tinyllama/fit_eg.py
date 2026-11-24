"""calibrate.py's fit again, at a SiLU key exponent eg (decision 0094). Hallmark code, MIT.

Usage: python fit_eg.py TREE GGUF_DIR WSUMS MAXIMA_JSON E_RES EG OUT_WORDS_JSON

Upstream's scales/calibrate.py fits the scales with fit_scales.fit() at eg 8 and has no option for
another eg; fit_scales.py has --eg but reads a shape and exponents that no program writes
(fit_scales.py, the comment above main()). This program makes the same call as calibrate.py, from
the same inputs (the shape from gguf_weights.shape_words(), the exponents from
calibrate.exponents_for_the_fit() and the sums file, the maxima from calibrate.py's maxima.json)
with eg as given, and completes words.json as calibrate.py does ("ew", "emb_bits"). It changes no
upstream file. At eg 8 it gives calibrate.py's words.json byte for byte. Exit status: that of
fit_scales.report(), 0 if words_to_policy.py will take the file.
"""

import json
import sys


def main(tree, gguf_dir, wsums, maxima_path, e_res, eg, out):
    sys.path.insert(0, f"{tree}/scales")
    import calibrate  # noqa: PLC0415 - upstream's, from the tree
    import fit_scales  # noqa: PLC0415
    import gguf_weights  # noqa: PLC0415

    shape = gguf_weights.shape_words(gguf_weights.open_model(gguf_dir))
    sums = calibrate.read_sums(wsums)
    made_at, embedding_bits = calibrate.line_of(sums, calibrate.EMBEDDING)
    if made_at != int(e_res):
        raise SystemExit(f"the sums file was made with --eres {made_at}, not {e_res}")
    exponents = calibrate.exponents_for_the_fit(sums, shape)
    with open(maxima_path) as f:
        maxima = json.load(f)
    words, notes = fit_scales.fit(shape, maxima, exponents, e_res=int(e_res), eg=int(eg))
    words["ew"] = {
        name: exponent for name, (exponent, _) in sums.items() if name != calibrate.EMBEDDING
    }
    words["emb_bits"] = embedding_bits
    fit_scales.write_private_json(out, words)
    return fit_scales.report(notes)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
