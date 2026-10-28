# fz-temporal-decision-visionai

Reference code for **FZ-Temporal-Decision-VisionAI-2B**, a 2B vision-language model that decides whether a CCTV view shows a
real fire or fire smoke by looking at **two moments at once**: the current frame and the same view about one second
earlier, packed into the temporal patch slot of the Qwen2-VL vision encoder (no extra vision compute).

Paper: *Did It Change? Two-Frame Temporal Verification Makes a 2B Vision-Language Model a Practical Fire and Smoke
Alarm on Edge NPUs* (Thanh Nguyen, Fizibox Research Team, Vietnam, 2026), preprint v1.1
[doi:10.5281/zenodo.23661072](https://doi.org/10.5281/zenodo.23661072) (all versions; v1.1:
[10.5281/zenodo.23726336](https://doi.org/10.5281/zenodo.23726336)); arXiv link added on announcement.
Model: [Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B](https://huggingface.co/Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B).

## Results in the paper (unseen data)

| | Transition videos (10 incidents) | 25-incident compilation | Real webcams, 47 h |
|---|---|---|---|
| Fizibox product (this model + alarm logic), H100 | 10/10, 1 false alarm | 23-24/25, 0 | 0 alarms |
| Same, quantised on a Qualcomm QCS8550 NPU | 9/10, 1 | | 0 alarms (full replay on the device) |
| Base Qwen2-VL-2B, Intel's yes/no recipe, as published | 9/10, 1 | 25/25, 16 | yes on 5.4-43.9% of scans |
| Same, with 3-of-4 confirmation | 8/10, 0 | 24/25, 12 | in alarm 1.7-41.5% of the time |
| WeVisOne-VL-4B, detection | 10/10, 4 | 25/25, 10 | 51-62% of frames positive |

The untuned 2B model finds incidents about as well, but cannot tell them from sunsets, lights, haze and fireworks.
On the NPU one image pair with both questions takes a median 274 ms; on an Intel Xeon CPU about 0.2 s with OpenVINO.

**v1.1 (Oct 2026):** comparison with Intel's recipe, 47 h replay on the QCS8550, CPU runtimes (ONNX/OpenVINO),
note on JPEG re-compression of small crops.

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
huggingface-cli login
```

Model access is gated: accept the terms on the [model page](https://huggingface.co/Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B)
first, then log in so `infer.py` can download the weights. ONNX Runtime and OpenVINO exports:
[FZ-Temporal-Decision-VisionAI-2B-ONNX-OpenVINO](https://huggingface.co/Fizibox-Corp/FZ-Temporal-Decision-VisionAI-2B-ONNX-OpenVINO).

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
