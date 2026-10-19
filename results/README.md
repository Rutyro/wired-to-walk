# Results

Work-in-progress visual material. These files describe the task and a short
diagnostic run; they are not a demonstrated learning result.

## `task-overview.png`

A plain-language diagram of the tethered fly-on-ball task: the 273 sensor
values, the 59 actuator commands, and the forward-speed reward.

## `training-20-updates.png`

PPO diagnostic run: 20 updates, 8 environments, 64 rollout steps per
environment, 4 epochs. The deterministic evaluation reward moved from
`0.436` to `0.937`, but the rollout reward stayed noisy, so this is a
pipeline check rather than evidence that the policy learned to walk.

Regenerate the plot from the committed report:

```powershell
python -I scripts/plot_ppo.py results/ppo-20-updates.json results/training-20-updates.png
```
