# Class Label Alignment

## Unified Definition
- `0 = non-falling`
- `1 = falling`

## Old v2 Difference
- old `v2` offline training used `0 = falling`, `1 = non-falling`
- `online_train` intentionally breaks that compatibility to match the current deployment-side definition

## Online Alert Condition
- `prediction == 1` -> `ALERT`
- `prediction == 0` -> `NORMAL`

## Design-Doc Note
- `lcfall_ros2_design_final.md` line 57 already states `prediction == falling -> ALERT`
- line 365 currently uses `if msg.prediction == 1:`
- this only remains correct if the online implementation also uses `1 = falling`
