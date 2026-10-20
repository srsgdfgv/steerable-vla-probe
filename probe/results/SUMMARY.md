# Result summary (frozen from bundled CSVs)

Scope: 32 Bridge trajectories, OpenVLA Steerable checkpoint `bridge_orig`. Numbers below are from `probe.csv`, `grip_greedy.csv`, `trace_shift.csv` in this folder. Re-run `python analyze.py` to regenerate tables.

## Main probe (`probe.csv`, 7464 rows)

Median xyz KL and ratio to same-style baseline median:

| Perturbation | Median xyz KL | Same-style baseline median | Ratio |
| --- | ---: | ---: | ---: |
| atomic / dir | 0.094 | 0.004 | ~26× |
| language / name | 0.014 | 0.002 | ~7× |
| point / name | 0.014 | 0.006 | ~2× |
| point / both | 0.018 | 0.006 | ~3× |
| point / coord | 0.001 | 0.006 | ~0.1× |
| trace / coord (mirror) | 0.005 | 0.005 | ~1× |

Same-frame paired rate P(edit > baseline) xyz (approx.):

| Edit | P(edit > baseline) |
| --- | ---: |
| dir | 0.84 |
| language name | 0.72 |
| point name | 0.59 |
| point both | 0.63 |
| point coord | 0.19 |
| trace coord | 0.25 |

Other notes:

- Rot medians ≤ ~0.002 for all variants.
- Point frames with both name and coord edits: name > coord on ~88% of frames.
- Dir effect is larger for open/close-only strings than for motion words alone (see analysis scripts historically); grip channel under teacher forcing stays near zero entropy on most frames.

## Grip free decode (`grip_greedy.csv`, 445 frames)

| Quantity | Value |
| --- | ---: |
| Frames where grip output changes | 27% (119 / 445) |
| Trajectories with ≥1 grip change | 90% |
| Trajectories with grip change on >50% of frames | 10% |
| Frames with no grip change and xyz L2 ≤ 0.001 | 45% |

When grip does flip, open↔close tends to swap the near-binary grip value (0 vs ~1). Most frames ignore the word flip.

## Trace horizontal shift (`trace_shift.csv`)

Global baseline median xyz KL ≈ 0.005.

| Shift | n | Median edit KL | P(edit > same-frame baseline) |
| --- | ---: | ---: | ---: |
| ±40 px | 1518 | 0.0016 | 0.10 |
| ±80 px | 788 | 0.0024 | 0.24 |
| ±100 px | 420 | 0.0017 | 0.16 |

Large translation does **not** systematically beat paraphrase noise under this metric.

## Claims boundary

**Supported:** under this offline, in-distribution Bridge probe, direction words and object names move the action distribution more than pixel coordinates; coordinates usually look like paraphrase-level noise.

**Not supported:** “model never learned coordinates,” “paper is wrong,” closed-loop failure of coordinate instructions, or Agent routing recommendations without further tests.
