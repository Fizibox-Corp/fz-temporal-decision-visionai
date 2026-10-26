"""Two-frame fire/smoke verification with FZ-Verdict-2B-Temporal.

The earlier frame goes into temporal slot 0 of the Qwen2-VL vision patch, the current frame into slot 1.
Each question is answered by the softmax over its option letters at the answer position.

  python infer.py --prev a.jpg --cur b.jpg
  python infer.py --video clip.mp4 --step 1.0 --out clip.jsonl
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DEFAULT_MODEL = "Fizibox-Corp/FZ-Verdict-2B-Temporal"
SIZE = 224
LETTERS = "ABC"

QUESTIONS = {
    "verdict": {
        "text": "Is there a real, uncontrolled fire, or smoke coming from a fire, in this scene? Lamps, vehicle "
                "or emergency lights, reflections, sunsets, screens, steam, fog, haze and dust are NOT fire or smoke.",
        "calib_key": "verdict_strict",
        "options": {"fire": "FIRE: active flames are visible",
                    "smoke": "SMOKE: smoke is visible but no flames",
                    "safe": "SAFE: no fire and no smoke"},
    },
    "motion": {
        "text": "This camera view combines an earlier moment and the current moment of the same place. Between the "
                "two moments, is there smoke or flame that moves, flickers or grows?",
        "calib_key": "motion",
        "options": {"yes": "Yes, smoke or flame changes between the two moments",
                    "no": "No, nothing smoke- or fire-like changes"},
    },
}


def prompt_text(q):
    opts = [f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"].values())]
    return "\n".join([f"Question: {q['text']}", "Options:", *opts, "Answer with the letter only."])


def to_square(img, roi=None):
    """Optional ROI (x, y, w, h) in pixels, then centre square, resized to 224x224."""
    img = img.convert("RGB")
    if roi:
        x, y, w, h = roi
        img = img.crop((x, y, x + w, y + h))
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((SIZE, SIZE), Image.BICUBIC)


class Verifier:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        path = model if os.path.isdir(model) else self._download(model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()
        calib = os.path.join(path, "calib.json")
        self.temps = json.load(open(calib))["temperatures"] if os.path.exists(calib) else {}
        tok = self.processor.tokenizer
        self.letter_ids = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in LETTERS],
                                       device=self.device)

    @staticmethod
    def _download(repo):
        from huggingface_hub import snapshot_download
        return snapshot_download(repo)

    def _encode(self, prev, cur, prompts):
        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            add_generation_prompt=True, tokenize=False) for p in prompts]
        enc = self.processor(text=texts, images=[cur] * len(prompts), padding=True, return_tensors="pt")
        pv_prev = self.processor.image_processor(images=[prev] * len(prompts), return_tensors="pt")["pixel_values"]
        pv = enc["pixel_values"]
        n, d = pv.shape
        t = self.processor.image_processor.temporal_patch_size
        a, b = pv.view(n, 3, t, -1), pv_prev.view(n, 3, t, -1)
        a[:, :, 0] = b[:, :, 0]  # earlier frame -> temporal slot 0, current frame stays in slot 1
        enc["pixel_values"] = a.reshape(n, d)
        return enc

    @torch.no_grad()
    def __call__(self, prev, cur, roi=None):
        """prev, cur: PIL images of the same camera about 1 s apart. Returns calibrated probabilities."""
        prev, cur = to_square(prev, roi), to_square(cur, roi)
        prompts = [prompt_text(q) for q in QUESTIONS.values()]
        enc = {k: v.to(self.device) for k, v in self._encode(prev, cur, prompts).items()}
        z = self.model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()[:, self.letter_ids]
        out = {}
        for i, (name, q) in enumerate(QUESTIONS.items()):
            k = len(q["options"])
            p = torch.softmax(z[i, :k] / float(self.temps.get(q["calib_key"], 1.0)), -1).tolist()
            out[name] = {key: round(v, 4) for key, v in zip(q["options"], p)}
        return out


def video_pairs(path, step, gap):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if ok else None

    t = gap
    while t * fps < n:
        prev, cur = frame(t - gap), frame(t)
        if prev is None or cur is None:
            break
        yield round(t, 2), prev, cur
        t += step
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local directory")
    ap.add_argument("--prev", help="earlier frame (about 1 s before --cur)")
    ap.add_argument("--cur", help="current frame")
    ap.add_argument("--video", help="video file; scores one pair every --step seconds")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between the two frames of a pair")
    ap.add_argument("--roi", help="optional region x,y,w,h in pixels")
    ap.add_argument("--out", help="JSONL output for --video (default: stdout)")
    a = ap.parse_args()
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    v = Verifier(a.model)
    if a.video:
        f = open(a.out, "w") if a.out else sys.stdout
        name = os.path.basename(a.video)
        for t, prev, cur in video_pairs(a.video, a.step, a.gap):
            r = v(prev, cur, roi)
            f.write(json.dumps({"source": name, "t": t, **r["verdict"], "motion": r["motion"]["yes"]}) + "\n")
            f.flush()
    elif a.prev and a.cur:
        print(json.dumps(v(Image.open(a.prev), Image.open(a.cur), roi), indent=1))
    else:
        ap.error("give --prev and --cur, or --video")


if __name__ == "__main__":
    main()
