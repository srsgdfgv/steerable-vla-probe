"""Action distribution helpers for the offline steerability probe.

Depends on the upstream OpenVLA / Prismatic stack:
  https://github.com/steerable-policies/steerable-policies-bridge
"""
import glob
import json
import sys
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from prismatic.models import load_vla
from config import ANN_DIR, CKPT_DIR, EP_DIR

CKPT = next(iter(glob.glob(f"{CKPT_DIR}/*.pt")), None)
UNNORM = "bridge_orig"
N_BINS = 256
GROUPS = {"xyz": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}


def load():
    vla = load_vla(CKPT, load_for_training=False)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    return vla.to("cuda").eval()


def prep(img):
    """Match the RLDS training resize: lanczos3 to 224."""
    img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
    return Image.fromarray(tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy())


def prompt_ids(vla, instruction):
    pb = vla.get_prompt_builder()
    pb.add_turn(role="human", message=instruction.lower().strip().rstrip(".") + ".")
    ids = vla.llm_backbone.tokenizer(pb.get_prompt(), truncation=True, return_tensors="pt").input_ids
    if ids[0, -1] != 29871:
        ids = torch.cat([ids, torch.tensor([[29871]])], dim=1)
    return ids


def predict(vla, image, instruction):
    """Greedy action: (continuous 7-d action, 7 action tokens)."""
    action, gen = vla.predict_action(
        image, instruction, unnorm_key=UNNORM, add_period_at_end=True, do_sample=False
    )
    return action, torch.as_tensor(gen[0, -7:])


@torch.inference_mode()
def action_probs(vla, image, instruction, action_ids):
    """Teacher-forced per-dim 256-bin distributions, shape (7, 256)."""
    ids = torch.cat([prompt_ids(vla, instruction), action_ids[None].cpu()], dim=1).to(vla.device)
    pixel = vla.vision_backbone.get_image_transform()(image)
    if isinstance(pixel, dict):
        pixel = {k: v[None].to(vla.device) for k, v in pixel.items()}
    else:
        pixel = pixel[None].to(vla.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = vla(input_ids=ids, pixel_values=pixel).logits[0, -8:-1]
    V = vla.llm_backbone.tokenizer.vocab_size
    return logits[:, V - N_BINS:V].float().softmax(-1).cpu().numpy()


def entropy(p):
    """Entropy of the original-instruction action distribution, grouped."""
    h = -(p * np.log(p + 1e-10)).sum(-1)
    return {f"ent_{g}": float(h[s].sum()) for g, s in GROUPS.items()}


def kl(p, q):
    """Per-dim KL(p || q), summed into xyz / rot / grip."""
    d = (p * (np.log(p + 1e-10) - np.log(q + 1e-10))).sum(-1)
    return {g: float(d[s].sum()) for g, s in GROUPS.items()}


if __name__ == "__main__":
    # Usage: python probe/action_dist.py <traj> <frame> "ins1" "ins2" ...
    traj, frame, instructions = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    image = prep(np.load(f"{EP_DIR}/{traj}.npz")["images"][frame])
    sub = json.load(open(f"{ANN_DIR}/step_to_subtask_dict.json"))[traj][str(frame)]
    print(f"traj {traj} frame {frame} subtask: {sub}\n")

    vla = load()
    np.set_printoptions(precision=4, suppress=True)
    base, base_ids = predict(vla, image, instructions[0])
    p0 = action_probs(vla, image, instructions[0], base_ids)
    V = vla.llm_backbone.tokenizer.vocab_size
    assert (p0.argmax(-1) + V - N_BINS == base_ids.numpy()).all(), "action_probs != predict_action"

    for ins in instructions:
        action, _ = predict(vla, image, ins)
        d = kl(p0, action_probs(vla, image, ins, base_ids))
        print(f"ins: {ins}\n  action: {action}")
        print(f"  KL vs first: xyz {d['xyz']:.3f}  rot {d['rot']:.3f}  grip {d['grip']:.3f}\n")
