- **A cloned model rescaled a resized compartment's species wrongly.**
  `Model.clone()` recompiled each event's expressions into the clone's own
  evaluator except the compartment size a resize rescale reads (added for
  #936), so every clone read some other expression as the size: `run_batch`
  rows (sequential and parallel) and `Model.clone()` gave a wrong
  concentration after an event resized a compartment (12.33 where 7.89 is
  right), and a clone could refuse a forward-sensitivity run the original
  accepts.
