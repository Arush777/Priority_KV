# PriorityBench W3 lock audit

- **Manifest:** `data/prioritybench/manifests/w3_lock.json`
- **SHA256:** `fc44b966725738c94008ba61ce57ad7366169b9c0be73074f8161d909ccfae89`
- **n:** 240 (expect 240)
- **category_hist:** {'tool_schema': 80, 'instruction_supersession': 80, 'multi_turn_state': 80}
- **split_hist:** {'calibration': 92, 'test': 99, 'validation': 49}
- **context_hist:** {8000: 83, 16000: 81, 32000: 76}
- **w2d_preserved:** 145
- **buried_by_cat:** `{"tool_schema": {"buried": 0, "plain": 80, "n": 80}, "instruction_supersession": {"buried": 20, "plain": 60, "n": 80}, "multi_turn_state": {"buried": 20, "plain": 60, "n": 80}}`
- **synth_selfcheck_errors:** 0
- **PASS:** True

## Notes

- Pool-level buried target is **25% (20/80)** per category where room exists;
  `tool_schema` may be 0 buried when W2d preserve already fills the quota.
- Every example carries `meta.buried_state` for slice reporting.
- Locked test examples must not be retuned after this SHA256 line is written
  to `FINAL_RUN_MANIFEST.yaml`.
