# PriorityBench W3 dual audit (15%)

- **Manifest:** `data/prioritybench/manifests/w3_lock.json`
- **SHA256 (must match lock):** `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89`
- **Sample:** n=36 / 240 (15%) seed=20260815
- **category_hist:** {'instruction_supersession': 10, 'multi_turn_state': 15, 'tool_schema': 11}
- **split_hist:** {<Split.TEST: 'test'>: 20, <Split.CALIBRATION: 'calibration'>: 10, <Split.VALIDATION: 'validation'>: 6}
- **dual_synth_errors:** 0
- **PASS:** True

## Method

Deterministic 15% sample; independent re-validation of shape + synth gold scoring
(same synth path as `audit_bench.py`). Does **not** retune locked examples.

All sampled examples passed second-pass synth scoring.
