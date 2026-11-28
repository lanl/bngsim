"""The coin flips of #1914's family cases and #1915's coincident surfaces and
events: points whose outcome changes as the step sequence moves by an ulp or so,
left out of the sweep and the slice with their counts (issues #1914 and #1915,
phase 0 of #1781). #1915's are below #1914's, with their own count of runs.

Every point of the 14 family cases was run 263 times: at the committed atol,
and at ``atol_ratio*(1 + d)`` for 262 values of d up to 2.5e-11 in size (24
spread over |d| <= 1e-12, 23 from 1e-12 to 2.3e-11, 192 at
+-(m*0.2603e-12 + 0.0137e-12) for m = 1..96, and the review's 23). Each such
change moves the step sequence and nothing else. A point here changed outcome
over those runs, or passed all of them but on one rose to within 2 times its
allowance from a median at most two thirds of that: an excursion of #1997 or
#1999 that another step sequence may carry past it. Every other point kept its
outcome on every run, every mark failing and every other point passing. Most
are #1799's short row, which shows on some step sequences only; the count is how
many of the 263 passed.

The list is what those runs found, not a bound: a point outside it can still
change outcome on another step sequence, at about 1 run in 250 or rarer for the
kinds here. A single new row of these kinds in the nightly's diff, after a
change that moves the step sequence, is that and not a regression.
"""

from __future__ import annotations

#: How many runs each point was read over.
RUNS = 263

#: Why the failing runs failed, by kind.
WHY = {
    "short": "on the runs that failed, its row on a stop came back short of its time (#1799)",
    "plain1996": "on the runs that failed, it failed as #1996 does",
    "sens_cverr": "on the run that failed, CVODE gave out with CV_ERR_FAILURE just past the step",
    "abs": (
        "on the runs that failed, the kd column at the corner came back about 1e-7 of itself off "
        "(#1999)"
    ),
    "moved": "on the runs that failed, #1998's moved state sat on or beside a crossing",
    "lost": (
        "on the run that failed, the w column's jump at the trailing edge was lost, -1010.9 for 0 "
        "from there to t = 30 (#2000, which may be #1984's kind)"
    ),
    "short_past": (
        "on the runs that failed, its row on the step came back short of its time (#1799), holding "
        "the column past the step: dX/dk_clock = -9.23 for 0"
    ),
}

#: key: (runs passed, a kind of ``WHY`` or the reason itself).
FLIPS = {
    "abs_state[kd=0.05,th=5.0]|rtol=1e-10,grid=aligned31,sens=kx+th+kd": (261, "abs"),
    "abs_state[kd=0.05,th=5.0]|rtol=1e-10,grid=offset31,sens=kx+th+kd": (261, "abs"),
    "abs_state[kd=2.0,kx=0.7]|rtol=1e-10,grid=aligned2,sens=kx+th+kd": (258, "abs"),
    "abs_state[kd=2.0,kx=0.7]|rtol=1e-10,grid=aligned31,sens=kx+th+kd": (258, "abs"),
    "abs_state[kd=2.0,kx=0.7]|rtol=1e-10,grid=offset31,sens=kx+th+kd": (261, "abs"),
    "gauss_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (
        156,
        "moved",
    ),
    "gauss_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (
        156,
        "moved",
    ),
    "gauss_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=u31,sens=kx+k_clock+c+w": (
        156,
        "moved",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-08,grid=aligned2,sens=kx+k_clock+c+w": (
        261,
        "short",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (
        259,
        "short",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=aligned31,sens=kx+k_clock+c+w": (
        261,
        "short",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=offset31,sens=kx+k_clock+c+w": (
        262,
        "lost",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (
        168,
        "moved",
    ),
    "pulse_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=u4,sens=none": (12, "plain1996"),
    "pulse_clock[c=50.74]|rtol=1e-12,grid=u4,sens=none": (21, "plain1996"),
    "sawtooth_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-06,grid=aligned2,sens=kx+k_clock+c+w": (
        258,
        "short",
    ),
    "sawtooth_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-06,grid=aligned31,sens=kx+k_clock+c+w": (
        257,
        "short",
    ),
    "sawtooth_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned2,sens=kx+k_clock+c+w": (
        245,
        "short",
    ),
    "sawtooth_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (
        253,
        "short",
    ),
    "sawtooth_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (
        168,
        "moved",
    ),
    "sawtooth_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=u4,sens=none": (12, "plain1996"),
    "sawtooth_clock[c=50.74]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (147, "moved"),
    "sawtooth_clock[c=50.74]|rtol=1e-12,grid=u4,sens=none": (21, "plain1996"),
    "sawtooth_clock[w=4.0]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (250, "short"),
    "sawtooth_clock[w=4.0]|rtol=1e-10,grid=aligned31,sens=kx+k_clock+c+w": (254, "short"),
    "sawtooth_clock[w=4.0]|rtol=1e-10,grid=offset31,sens=kx+k_clock+c+w": (256, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-08,grid=aligned2,sens=kx+k_clock+c+w": (261, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-08,grid=aligned31,sens=kx+k_clock+c+w": (261, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (260, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-10,grid=aligned31,sens=kx+k_clock+c+w": (260, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-12,grid=aligned2,sens=kx+k_clock+c+w": (262, "short"),
    "sawtooth_coupled[a=1.0,b=-1.0]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (262, "short"),
    "sawtooth_coupled[w=0.1]|rtol=1e-06,grid=aligned2,sens=kx+k_clock+c+w": (262, "short"),
    "sawtooth_coupled[w=0.1]|rtol=1e-06,grid=aligned31,sens=kx+k_clock+c+w": (262, "short"),
    "sawtooth_coupled|rtol=1e-12,grid=u4,sens=none": (12, "plain1996"),
    "step_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=aligned2,sens=k0+k1+k_clock+c": (
        262,
        "short",
    ),
    "step_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned2,sens=k0+k1+k_clock+c": (
        228,
        "short",
    ),
    "step_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned31,sens=k0+k1+k_clock+c": (
        256,
        "short",
    ),
    "step_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=offset31,sens=k0+k1+k_clock+c": (
        251,
        "short",
    ),
    "step_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=u31,sens=k0+k1+k_clock+c": (
        259,
        "short_past",
    ),
    "step_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=offset31,sens=k0+k1+k_clock+c": (259, "short"),
    "step_clock[c=200.0,k_clock=8.0]|rtol=1e-12,grid=aligned2,sens=k0+k1+k_clock+c": (
        262,
        "short",
    ),
    "step_clock[c=200.0,k_clock=8.0]|rtol=1e-12,grid=offset31,sens=k0+k1+k_clock+c": (
        261,
        "short",
    ),
    "step_clock[c=50.74]|rtol=1e-12,grid=aligned2,sens=k0+k1+k_clock+c": (262, "short"),
    "step_clock[k1=30.0]|rtol=1e-10,grid=aligned2,sens=k0+k1+k_clock+c": (262, "short"),
    "step_clock[k1=30.0]|rtol=1e-10,grid=offset31,sens=k0+k1+k_clock+c": (262, "short"),
    "step_clock[k1=30.0]|rtol=1e-10,grid=u4,sens=k0+k1+k_clock+c": (262, "sens_cverr"),
    "step_clock|rtol=1e-12,grid=offset31,sens=k0+k1+k_clock+c": (262, "short"),
    "triangle_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (
        256,
        "short",
    ),
    "triangle_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-10,grid=offset31,sens=kx+k_clock+c+w": (
        257,
        "short",
    ),
    "triangle_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (
        147,
        "moved",
    ),
    "triangle_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (
        147,
        "moved",
    ),
    "triangle_clock[c=12.5,k_clock=0.5]|rtol=1e-12,grid=u31,sens=kx+k_clock+c+w": (147, "moved"),
    "triangle_clock[c=200.0,k_clock=8.0]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (
        262,
        "short",
    ),
    "triangle_clock[c=200.0,k_clock=8.0]|rtol=1e-10,grid=aligned31,sens=kx+k_clock+c+w": (
        262,
        "short",
    ),
    "triangle_clock[c=50.74]|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (259, "short"),
    "triangle_clock[w=0.1]|rtol=1e-12,grid=aligned2,sens=kx+k_clock+c+w": (260, "short"),
    "triangle_clock[w=0.1]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (262, "short"),
    "triangle_clock[w=0.1]|rtol=1e-12,grid=offset31,sens=kx+k_clock+c+w": (258, "short"),
    "triangle_clock[w=4.0]|rtol=1e-12,grid=aligned2,sens=kx+k_clock+c+w": (262, "moved"),
    "triangle_clock[w=4.0]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (262, "moved"),
    "triangle_clock[w=4.0]|rtol=1e-12,grid=u31,sens=kx+k_clock+c+w": (262, "moved"),
    "triangle_clock|rtol=1e-10,grid=aligned2,sens=kx+k_clock+c+w": (259, "short"),
    "triangle_clock|rtol=1e-10,grid=offset31,sens=kx+k_clock+c+w": (262, "short"),
    # Near misses: every run passed, one came close.
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=u31,sens=kx+k_clock+c+w": (
        263,
        "on one run the w column past the trailing edge came back 4.1e-6 off where it is 0 "
        "(#1997), 0.80 of the allowance, against a median of 0.31",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned2,sens=kx+k_clock+c+w": (
        263,
        "on one run the w column on the trailing edge came back 2.7e-6 off where it is 0 "
        "(#1997), 0.54 of the allowance, against a median of 0.31",
    ),
    "pulse_clock[T0=-49.95,c=0.04999999999999716]|rtol=1e-12,grid=aligned31,sens=kx+k_clock+c+w": (
        263,
        "on one run the w column on the trailing edge came back 3.3e-6 off where it is 0 "
        "(#1997), 0.65 of the allowance, against a median of 0.31",
    ),
    "abs_state[kd=2.0,kx=0.7]|rtol=1e-12,grid=offset31,sens=kx+th+kd": (
        263,
        "on one run the kd column beside the corner came back 854 times its tolerance (#1999), "
        "0.85 of the allowance, against a median of 20",
    ),
}


# ── #1915: coincident surfaces and events ───────────────────────────────────
#
# Every point of #1915's seven cases, exclusions included, was run 480 times: at
# the committed atol and at ``atol_ratio*(1 + d)`` for 239 values of d on a grid
# up to 2.5e-11 in size (24 spread over |d| <= 1e-12, 192 at
# +-(m*0.2603e-12 + 0.0137e-12) for m = 1..96, and 23 from 1e-12 to 2.5e-11
# alternating in sign), and for 240 drawn at random (seed 4242): half uniform in
# +-2.5e-11, half log-uniform in size from 1e-14 to 2.5e-11. The rule is #1914's,
# and so is the caveat: the list is what those runs found, not a bound. The
# random set found 13 points the grid had not; a review's 100 further draws
# found 5 more, and its next 220 found 4 more, each failing or coming close once
# there (counted below as passing all 480 of these runs). Across a whole sweep
# of these cases, between 1 run in 25 and 1 in 220 shows one such point.

#: How many runs each #1915 point was read over.
RUNS_1915 = 480

WHY.update(
    {
        "residue": (
            "on the runs that failed, a column past the clock pulse came back off where it is 0 "
            "by more than the allowance (#1997's residue, which these rows carry at rtol 1e-12 on "
            "every run)"
        ),
        "located": (
            "on the runs that failed, a crossing on the state was located off in time, about 1e-8 "
            "at rtol 1e-12 and up to 1.7e-7 in a sensitivity run at rtol 1e-10, and the species "
            "near it came back off by more than the allowance (#2005)"
        ),
        "review_located": (
            "passed all 480 runs here, but failed on one of the review's: a crossing on the state "
            "located off in time, and the species near it off by more than the allowance (#2005)"
        ),
        "review_short": (
            "passed all 480 runs here, but failed on one of the review's: its row on the crossing "
            "came back short of its time (#1799)"
        ),
        "short_edge": (
            "on the runs that failed, its row on the pulse's leading edge came back short of its "
            "time (#1799), holding the columns past the edge: dX/dkc = -18.75 or dX/dc = 1.25 for 0 "
            "at t = 25"
        ),
        "near816": (
            "passed all 480 runs here, but on one of the review's at 816 times its tolerance, 0.82 "
            "of the allowance, against a median of 52"
        ),
        "near801": (
            "passed all 480 runs here, but on one of the review's at 801 times its tolerance, 0.80 "
            "of the allowance, against a median of 23"
        ),
    }
)

FLIPS_1915 = {
    "coincident_state[th1=6.0,th2=6.0]|rtol=1e-10,grid=aligned2,sens=k0+d1+d2": (479, "short"),
    "coincident_state[th1=6.0,th2=6.0]|rtol=1e-10,grid=aligned31,sens=k0+d1+d2": (479, "short"),
    "coincident_state[th1=6.0,th2=6.0]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2": (471, "short"),
    "coincident_state[th1=6.0,th2=6.0]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2": (471, "short"),
    "coincident_state[th2=1.0]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2": (480, "review_short"),
    "coincident_state[th2=1.0]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2+th1+th2": (
        480,
        "review_short",
    ),
    "coincident_state[th2=1.0]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2": (479, "short"),
    "coincident_state[th2=1.0]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2+th1+th2": (479, "short"),
    "coincident_state[th2=2.7]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2": (476, "short"),
    "coincident_state[th2=2.7]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2+th1+th2": (476, "short"),
    "coincident_state[th2=2.7]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2": (476, "short"),
    "coincident_state[th2=2.7]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2+th1+th2": (476, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-10,grid=aligned31,sens=k0+d1+d2": (478, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-10,grid=aligned31,sens=k0+d1+d2+th1+th2": (478, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2": (478, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2+th1+th2": (478, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2": (479, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=aligned31,sens=k0+d1+d2+th1+th2": (479, "short"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=u4,sens=k0+d1+d2": (479, "located"),
    "coincident_state[th2=2.9997]|rtol=1e-12,grid=u4,sens=k0+d1+d2+th1+th2": (479, "located"),
    "coincident_state[th2=3.0000000000000004]|rtol=1e-12,grid=aligned2,sens=k0+d1+d2": (
        480,
        "review_located",
    ),
    "coincident_state[th2=3.0000000000000004]|rtol=1e-12,grid=aligned2,sens=none": (
        480,
        "on one run it rose to 773 times its tolerance, 0.77 of the allowance, against a median of 3",
    ),
    "event_clock_reset[c=120.0,kc=8.0]|rtol=1e-10,grid=aligned2,sens=kx+kc+c+w+tau": (
        456,
        "short",
    ),
    "event_clock_reset[c=120.0,kc=8.0]|rtol=1e-10,grid=aligned31,sens=kx+kc+c+w+tau": (
        455,
        "short",
    ),
    "event_clock_reset[c=120.0,kc=8.0]|rtol=1e-12,grid=aligned2,sens=none": (
        480,
        "passed all 480 runs here, but on one of the review's at 589 times its tolerance, 0.59 of the allowance, against a median of 57 here",
    ),
    "event_clock_reset[c=120.0,kc=8.0]|rtol=1e-12,grid=aligned31,sens=none": (
        480,
        "passed all 480 runs here, but on one of the review's at 589 times its tolerance, 0.59 of the allowance, against a median of 44 here",
    ),
    "event_clock_reset[c=120.0,kc=8.0]|rtol=1e-12,grid=u31,sens=none": (479, "located"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-08,grid=offset31,sens=kx+kc+c+w+tau": (
        478,
        "residue",
    ),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-08,grid=u31,sens=kx+kc+c+w+tau": (478, "residue"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-08,grid=u4,sens=kx+kc+c+w+tau": (391, "residue"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (
        479,
        "residue",
    ),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-10,grid=u31,sens=kx+kc+c+w+tau": (479, "residue"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-10,grid=u4,sens=kx+kc+c+w+tau": (479, "residue"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-12,grid=u31,sens=none": (479, "located"),
    "event_clock_reset[c=18.0,tau=16.0]|rtol=1e-12,grid=u4,sens=none": (479, "located"),
    "event_clock_reset[tau=10.37]|rtol=1e-08,grid=aligned2,sens=kx+kc+c+w+tau": (470, "short"),
    "event_clock_reset[tau=10.37]|rtol=1e-08,grid=aligned31,sens=kx+kc+c+w+tau": (476, "short"),
    "event_clock_reset[tau=10.37]|rtol=1e-10,grid=aligned2,sens=kx+kc+c+w+tau": (470, "short"),
    "event_clock_reset[tau=10.37]|rtol=1e-10,grid=aligned31,sens=kx+kc+c+w+tau": (469, "short"),
    "event_clock_reset[tau=10.37]|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (479, "located"),
    "event_clock_reset[tau=10.37]|rtol=1e-10,grid=u4,sens=kx+kc+c+w+tau": (480, "near816"),
    "event_clock_reset[tau=10.37]|rtol=1e-12,grid=offset31,sens=none": (480, "near801"),
    "event_clock_reset[tau=15.02]|rtol=1e-08,grid=aligned2,sens=kx+kc+c+w+tau": (298, "short"),
    "event_clock_reset[tau=15.02]|rtol=1e-10,grid=aligned31,sens=kx+kc+c+w+tau": (456, "short"),
    "event_clock_reset[tau=15.02]|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (456, "short"),
    "event_clock_reset[tau=15.02]|rtol=1e-12,grid=aligned2,sens=kx+kc+c+w+tau": (399, "short"),
    "event_clock_reset[tau=15.02]|rtol=1e-12,grid=aligned31,sens=kx+kc+c+w+tau": (298, "short"),
    "event_clock_reset[tau=15.02]|rtol=1e-12,grid=offset31,sens=kx+kc+c+w+tau": (298, "short"),
    "event_clock_reset[tau=20.0]|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (
        480,
        "on one run the w column past the pulse came back 1.6e-6 off where it is 0 (#1997's residue), 539 times its tolerance, 0.54 of the allowance, against a median of 55",
    ),
    "event_clock_reset[tau=20.0]|rtol=1e-10,grid=u31,sens=kx+kc+c+w+tau": (480, "review_located"),
    "event_clock_reset[tau=20.0]|rtol=1e-10,grid=u4,sens=kx+kc+c+w+tau": (479, "located"),
    "event_clock_reset[tau=20.0]|rtol=1e-12,grid=u4,sens=none": (
        480,
        "on one run its species rose to 527 times their tolerance, 0.53 of the allowance, against a median of 20",
    ),
    "event_clock_reset[w=4.0]|rtol=1e-08,grid=aligned2,sens=kx+kc+c+w+tau": (443, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-08,grid=aligned31,sens=kx+kc+c+w+tau": (455, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-08,grid=u31,sens=kx+kc+c+w+tau": (455, "short_edge"),
    "event_clock_reset[w=4.0]|rtol=1e-10,grid=aligned2,sens=kx+kc+c+w+tau": (421, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-10,grid=aligned31,sens=kx+kc+c+w+tau": (447, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (444, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-10,grid=u31,sens=kx+kc+c+w+tau": (447, "short_edge"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=aligned2,sens=kx+kc+c+w+tau": (281, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=aligned2,sens=none": (
        480,
        "on one run its species rose to 818 times their tolerance, 0.82 of the allowance, against a median of 20",
    ),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=aligned31,sens=kx+kc+c+w+tau": (113, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=offset31,sens=kx+kc+c+w+tau": (97, "short"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=u31,sens=kx+kc+c+w+tau": (314, "short_edge"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=u4,sens=kx+kc+c+w+tau": (479, "residue"),
    "event_clock_reset[w=4.0]|rtol=1e-12,grid=u4,sens=none": (477, "located"),
    "event_clock_reset|rtol=1e-08,grid=aligned2,sens=kx+kc+c+w+tau": (458, "short"),
    "event_clock_reset|rtol=1e-08,grid=aligned31,sens=kx+kc+c+w+tau": (449, "short"),
    "event_clock_reset|rtol=1e-10,grid=aligned2,sens=kx+kc+c+w+tau": (467, "short"),
    "event_clock_reset|rtol=1e-10,grid=aligned31,sens=kx+kc+c+w+tau": (462, "short"),
    "event_clock_reset|rtol=1e-10,grid=offset31,sens=kx+kc+c+w+tau": (
        480,
        "passed all 480 runs here, but on one of the review's at 578 times its tolerance, 0.58 of the allowance, against a median of 50 here",
    ),
    "event_clock_reset|rtol=1e-10,grid=u31,sens=kx+kc+c+w+tau": (475, "residue"),
    "event_clock_reset|rtol=1e-10,grid=u4,sens=kx+kc+c+w+tau": (
        480,
        "on one run it rose to 695 times its tolerance, 0.70 of the allowance, against a median of 49",
    ),
}

#: Every case's flips, by key.
ALL = {
    **{k: (p, w, RUNS) for k, (p, w) in FLIPS.items()},
    **{k: (p, w, RUNS_1915) for k, (p, w) in FLIPS_1915.items()},
}


def for_case(name: str) -> dict[str, str]:
    """The case's flips as ``Case.exclude``."""
    out = {}
    for k, (passed, kind, runs) in ALL.items():
        if k.split("|", 1)[0].split("[", 1)[0] != name:
            continue
        head = (
            f"a coin flip on the step sequence: passed {passed} of {runs} runs"
            if passed < runs
            else f"rides on the step sequence: passed all {runs} runs"
        )
        out[k] = f"{head} whose atol differed by up to 2.5 parts in 1e11; {WHY.get(kind, kind)}"
    return out
