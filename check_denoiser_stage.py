"""
check_denoiser_stage.py — STAGE_SPEC §11 build-order step 4 checks, for
`model/denoiser.py` + `model/conditioning.py`.

Check 4 needs care (STAGE_SPEC's own §11 flags this: "check 4 fails
silently"): AdaLN-Zero's whole point is that every block starts as an
EXACT identity map regardless of conditioning — the final Linear in
`AdaLNZero.net` is zero-init (weight AND bias), so `shift/scale/gate`
are identically 0 for ANY `emb`, hence any `t`. So `model(x, t=0)` and
`model(x, t=999)` are expected to be BIT-IDENTICAL at random init — that
is not a bug, and "run it once at init, see identical outputs, conclude
broken" is exactly the silent-failure trap. The real check is whether
GRADIENT reaches the gates and grows them away from zero (LOGGING.md
panel 2) — tested here with a short training probe, not just a static
forward pass.

Run: python check_denoiser_stage.py
"""

import container_env  # noqa: F401 -- MUST precede torch, see its docstring

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn.functional as F

from batching import lengths_to_offset
from model.denoiser import Denoiser
from stage_config import StageConfig
from train.dataset import ChairPointDataset

torch.manual_seed(0)
# n_gaussians left at the production default (16384, not clipped down for
# speed): STAGE_SPEC §3's smallest objects have 11,100 < 16,384 Gaussians,
# so this is what actually produces genuine variable-N across objects --
# a smaller n_gaussians would clip every object to the same length and
# check 5 wouldn't be testing anything.
cfg = StageConfig(stage=1, num_workers=0)  # stage 1 on purpose: the checks
# below are about t-conditioning and variable-N batching, neither of which
# depends on which channels are the target, and stage 1 keeps these numbers
# comparable against the ones already recorded in PLAN.md.
model = Denoiser(cfg)

ds = ChairPointDataset(cfg, split="train", augment=False)

# ----------------------------------------------------------------- check 5: variable-length batching
print("[check 5] variable-length batching: per-object output must match a solo run")
model.eval()  # fixed serialization cycling per block (§4c), not random -- deterministic for this check

# Most objects exceed n_gaussians and clip to exactly cfg.n_gaussians, so
# picking the first 3 indices risks 3 equal lengths and testing nothing
# (hit this in-session with a smaller n_gaussians) -- scan for a genuinely
# mixed-length trio instead of assuming index order provides one.
samples, seen_lengths = [], set()
for i in range(len(ds)):
    s = ds[i]
    if s["n"] not in seen_lengths:
        samples.append(s)
        seen_lengths.add(s["n"])
    if len(samples) == 3:
        break
lengths = [s["n"] for s in samples]
print(f"    object lengths: {lengths}")
assert len(set(lengths)) > 1, "test setup: objects happen to have equal length, not a real test of variable-N"

cond_all = torch.cat([s["cond"] for s in samples], dim=0)
target_all = torch.cat([s["target"] for s in samples], dim=0)
offset_mixed = lengths_to_offset(lengths)
t_per_object = torch.tensor([100, 500, 900])  # distinct t per object -- realistic training condition

with torch.no_grad():
    out_mixed = model(cond_all, target_all, t_per_object, offset_mixed)

max_errs, cursor = [], 0
for i, s in enumerate(samples):
    n = s["n"]
    offset_solo = lengths_to_offset([n])
    with torch.no_grad():
        out_solo = model(s["cond"], s["target"], t_per_object[i:i + 1], offset_solo)
    out_from_mixed = out_mixed[cursor:cursor + n]
    max_err = (out_from_mixed - out_solo).abs().max().item()
    max_errs.append(max_err)
    print(f"    object {i} (n={n}): max|mixed - solo| = {max_err:.2e}")
    cursor += n

assert max(max_errs) < 1e-3, f"variable-length batching leaks across objects: max err {max(max_errs):.2e}"
print("    PASS\n")

# ----------------------------------------------------------------- check 4: timestep reaches the network
print("[check 4] does t reach the network? (see module docstring for why this needs two parts)")
model.eval()
probe = ds[3]
cond, offset = probe["cond"], lengths_to_offset([probe["n"]])
target_noisy = torch.randn_like(probe["target"])  # held fixed while only t varies

with torch.no_grad():
    out_t0 = model(cond, target_noisy, torch.tensor([0]), offset)
    out_t999 = model(cond, target_noisy, torch.tensor([999]), offset)
diff_init = (out_t0 - out_t999).abs().max().item()
print(f"    AT INIT:        max|model(t=0) - model(t=999)| = {diff_init:.2e}  "
      f"(expected exactly 0.0 -- AdaLN-Zero, see docstring)")
assert diff_init == 0.0, "t already affects output at init -- adaLN zero-init isn't working as designed"

print("    running a short training probe (200 steps) to confirm gradient reaches "
      "the gates and t starts to matter...")
model.train()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
probe_pool = [ds[i] for i in range(8)]
for step in range(200):
    s = probe_pool[step % len(probe_pool)]
    t_s = torch.randint(0, cfg.T, (1,))
    noise = torch.randn_like(s["target"])
    # stand-in noising just to exercise the gradient path -- the real DDPM
    # forward process (diffusion/schedule.py) is step 5, not built yet.
    alpha = 1.0 - t_s.item() / cfg.T
    noisy = (alpha ** 0.5) * s["target"] + ((1 - alpha) ** 0.5) * noise
    out = model(s["cond"], noisy, t_s, lengths_to_offset([s["n"]]))
    # Mean half only: under cfg.learn_sigma the model emits 2C channels and
    # the second half is the LEARNED_RANGE variance, not a prediction of the
    # target. This probe deliberately uses a plain MSE rather than the real
    # hybrid loss (diffusion/losses.py) -- all it needs is *some* gradient
    # reaching the adaLN gates, and a plain MSE keeps the check independent
    # of the loss it is not testing.
    pred = model.split_output(out)[0]
    loss = F.mse_loss(pred, s["target"])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 50 == 0:
        print(f"      step {step + 1}: loss={loss.item():.4f}")

model.eval()
gate_mags = [b.adaln.net[-1].weight.abs().mean().item() for b in model.backbone.blocks]
print(f"    adaLN gate weight |mean| per block after 200 steps:")
for i, g in enumerate(gate_mags):
    print(f"      block {i:2d}: {g:.2e}")
assert any(g > 0 for g in gate_mags), "gates did not move at all after 200 training steps"

with torch.no_grad():
    out_t0_trained = model(cond, target_noisy, torch.tensor([0]), offset)
    out_t999_trained = model(cond, target_noisy, torch.tensor([999]), offset)
diff_trained = (out_t0_trained - out_t999_trained).abs().max().item()
print(f"    AFTER 200 steps: max|model(t=0) - model(t=999)| = {diff_trained:.4f}")
assert diff_trained > 1e-4, "t=0 vs t=999 still identical after training -- adaLN gates not receiving gradient"
print("    PASS\n")

print("ALL DENOISER-STAGE CHECKS PASSED.")
