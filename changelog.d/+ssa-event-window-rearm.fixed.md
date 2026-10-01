- **SSA events no longer miss a trigger that is true only inside a time window,
  or a periodic trigger's later rises.** Triggers were looked at only at the end
  of each step: `time > 37.3 && time < 37.5` never fired, and `sin(time) > 0.9`
  fired once in (0, 50) instead of 8 times when nothing fired between its rises.
