# Verification and serving

Run the compatibility regression test after changing the model assembly or tactile path:

```bash
python scripts/gate_c_check.py
```

Additional validation utilities are available under `scripts/` for tactile causal dependence,
prefix-cache equivalence, and model compatibility.

## Serving

For the `n0_VTLA_univtac_single8` checkpoint on **Lift Can**, first read the
[wrist-camera compatibility note](LIFT_CAN_REPRODUCTION.md). UniVTAC `695a22d` changes the
camera unconditionally; setting `UNIVTAC_LIFT_CAN_FIXEDCAM=0` requires the upstream camera-switch
fix and must apply to the simulator process.

```bash
python scripts/serve_policy.py \
  --policy.config=vtla_tactile_posttrain \
  --policy.dir=checkpoints/vtla_tactile_posttrain/<experiment>/<step>
```

Robot-side clients can use `n0vtla_client`. Inputs must follow the canonical state,
camera, tactile-slot, and mask schema used during training.

The default action horizon is 50. Unless a policy is trained for a different replanning interval, execute the complete action chunk before requesting the next prediction.

## Further reading

- [GOTCHAS.md](GOTCHAS.md): implementation and optimization considerations.
- [TACTILE_CAUSAL_PROBE.md](TACTILE_CAUSAL_PROBE.md): how to measure whether the latent
  tactile token actually reads the tactile input, and how to read the result.
