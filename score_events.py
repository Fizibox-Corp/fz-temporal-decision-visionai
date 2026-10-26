"""Event-level scoring of per-frame fire/smoke scores against clip labels.

Scores: JSONL lines {"source": video, "t": seconds, "fire": p, "smoke": p, ...} (e.g. from infer.py --video).
Labels: CSV with columns video, t_start, t_end, label (FIRE/SMOKE/SAFE), onset (seconds, incidents only).

A frame is a hit when fire + smoke >= --th. An alarm starts when at least M of the last N frames of a clip are hits.
Incident clips: detected if an alarm starts at or after the onset; time-to-detect is measured from the onset; an
alarm more than 2 s before the onset counts as early. SAFE clips: alarm starts closer than --merge seconds are one
false-alarm episode, reported per hour.

  python score_events.py --labels labels.csv --scores clip.jsonl --th <threshold> [--m 3 --n 4]
"""
import argparse
import csv
import json
import statistics


def load_labels(path):
    clips = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        lab = r["label"].strip().upper()
        if lab not in ("FIRE", "SMOKE", "SAFE"):
            continue
        on = (r.get("onset") or "").strip()
        clips.append({"video": r["video"], "t0": float(r["t_start"]), "t1": float(r["t_end"]), "label": lab,
                      "onset": float(on) if on and lab != "SAFE" else None})
    return clips


def load_hits(path, th):
    by = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        by.setdefault(d["source"], []).append((d["t"], d["fire"] + d["smoke"] >= th))
    return {v: sorted(x) for v, x in by.items()}


def alarm_starts(timeline, m, n):
    hist, out, prev = [], [], False
    for t, hit in timeline:
        hist = (hist + [hit])[-n:]
        on = sum(hist) >= m
        if on and not prev:
            out.append(t)
        prev = on
    return out


def score(clips, hits, m, n, merge):
    ttd, missed, early, fa, safe_s, detail = [], 0, 0, 0, 0.0, []
    for c in clips:
        tl = [(t, h) for t, h in hits.get(c["video"], []) if c["t0"] <= t < c["t1"]]
        if not tl:
            continue
        starts = alarm_starts(tl, m, n)
        if c["label"] != "SAFE":
            on = c["onset"] if c["onset"] is not None else c["t0"]
            after = [t for t in starts if t >= on]
            early += any(t < on - 2.0 for t in starts)
            if after:
                ttd.append(after[0] - on)
            else:
                missed += 1
            detail.append({"video": c["video"], "t0": c["t0"], "label": c["label"],
                           "ttd": round(after[0] - on, 1) if after else None})
        else:
            safe_s += c["t1"] - c["t0"]
            last = -1e9
            for t in starts:
                if t - last > merge:
                    fa += 1
                    detail.append({"video": c["video"], "t0": c["t0"], "label": "FA", "t": t})
                last = t
    return {"incidents": len(ttd) + missed, "detected": len(ttd), "missed": missed,
            "ttd_median": round(statistics.median(ttd), 1) if ttd else None,
            "alarm_before_onset": early, "fa_episodes": fa, "safe_hours": round(safe_s / 3600, 3),
            "fa_per_hour": round(fa / (safe_s / 3600), 2) if safe_s else None, "detail": detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", required=True, action="append", help="JSONL; repeat to pool several files")
    ap.add_argument("--th", type=float, required=True)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--merge", type=float, default=30.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    hits = {}
    for p in a.scores:
        hits.update(load_hits(p, a.th))
    res = score(load_labels(a.labels), hits, a.m, a.n, a.merge)
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
