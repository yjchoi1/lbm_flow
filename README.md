# Fluid Flow Simulation with LBM 
## Velocity input options
### Uniform
`args = [min, max, nnodes_left]`
### Normal
`args = [...]`
### From `csv`
`velocities.csv`
```
     v0, v1, v2, ...
sim0
sim1
sim2
.
.
.
```

## Obstacle options
### From `json`
```json
{
  "sim0": [
    [40, 40, 5],
    [50, 20, 7],
    ...
  ],
  "sim1": [
    [40, 40, 5],
    [50, 20, 7],
    ...
  ],
  "sim2": [
    ...
  ],
  ...
}
```