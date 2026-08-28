"""Signal-processing core for running gait analysis.

Stage modules, in pipeline order:

    loader       -- stage 1: dataset access, time index, integrity/summary reporting
    orientation  -- stage 2: sensor -> anatomical frame resolution
    steps        -- stage 3: footstrike detection and cadence
    lateral      -- stage 4: exploratory left/right discrimination

`pipeline` glues them together for a single trial.

Design rules enforced throughout this package:

* Sample rate is a parameter (`fs_hz`) on every function that needs it.
  Nothing hardcodes 50 Hz. The dataset's nominal rate lives in exactly one
  place (`loader.DEFAULT_FS_HZ`) and is only ever a *default*.
* Every filter cutoff, threshold and window length is either derived from
  `fs_hz` / an estimated step frequency, or is a named module constant with
  a comment stating why that value.
* Algorithms are formulated to be placement-agnostic. Where a step
  nevertheless depends on the sensor sitting in a trouser pocket, it is
  flagged in the returned diagnostics rather than hidden.
"""

__all__ = ["loader", "orientation", "steps", "lateral", "pipeline", "plotting"]
