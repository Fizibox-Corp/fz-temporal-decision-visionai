# fz-temporal-decision-visionai

Reference code for **FZ-Temporal-Decision-VisionAI-2B**, a 2B vision-language model that decides whether a CCTV view shows a
real fire or fire smoke by looking at **two moments at once**: the current frame and the same view about one second
earlier, packed into the temporal patch slot of the Qwen2-VL vision encoder (no extra vision compute).

Paper: *Did It Change? Two-Frame Temporal Verification Makes a 2B Vision-Language Model a Practical Fire and Smoke
Alarm on Edge NPUs* (Thanh Nguyen, Fizibox Research Team, Vietnam, 2026), preprint
[doi:10.5281/zenodo.23661073](https://doi.org/10.5281/zenodo.23661073); arXiv link added on announcement.
Model: [Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B](https://huggingface.co/Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B).

## What the model answers

| Question | Options |
|---|---|
| `verdict` | fire / smoke / safe, ignoring named look-alikes (lamps, lights, reflections, sunsets, steam, fog, dust) |
| `motion` | yes / no: does any smoke or flame move, flicker or grow between the two moments? |

Each answer is the softmax over the option letters at the answer position, divided by a per-question temperature
from `calib.json`. No text is generated.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# one pair of frames, about 1 s apart
python infer.py --prev earlier.jpg --cur now.jpg

# a video: one pair per second -> JSONL (source, t, fire, smoke, safe, motion)
python infer.py --video clip.mp4 --step 1.0 --out clip.jsonl

# optional region of interest (x,y,w,h in pixels); the input is centre-cropped to a square and resized to 224x224
python infer.py --video clip.mp4 --roi 640,200,480,480 --out clip.jsonl
```

```python
from PIL import Image
from infer import Verifier

v = Verifier()  # downloads the weights from Hugging Face
print(v(Image.open("earlier.jpg"), Image.open("now.jpg")))
# {'verdict': {'fire': ..., 'smoke': ..., 'safe': ...}, 'motion': {'yes': ..., 'no': ...}}
```

`score_events.py` turns per-frame scores into alarms (M of the last N frames) and reports detection, time-to-detect
and false-alarm episodes per hour against clip labels.

## Scope

This repository returns the model's calibrated probabilities. The alarm logic of the Fizibox product (which image
regions are queried, how the answers are combined and confirmed over time) is not included, so results of this
reference code are not those reported for the product in the paper. Not a certified safety device.

## Licence

- Code in this repository: MIT (see `LICENSE`).
- Model weights: CC BY-NC 4.0, research and non-commercial use. No patent rights are granted. For commercial use
  contact thanh@fizibox.com.

## Citation

See `CITATION.cff`.
