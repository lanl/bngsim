- **A time threshold computed by an SBML assignment rule that reads state no
  longer gets a stop at a time nothing crosses.** `time >= g` with `g := 4*S` read
  the rule's parameter slot as a constant (4 at load) and clamped a step at t=4,
  while the crossing is at 1.705. The registered root still found the real
  crossing, so results were right, but the stop was spurious. The crossing
  resolver now inlines every assignment rule's body, as it already did for
  aliases of `time`: a body that reads state leaves the condition to its root,
  and one that reads parameters resolves on their live values.
