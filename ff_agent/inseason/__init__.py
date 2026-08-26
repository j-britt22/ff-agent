"""M10b — the in-season monitor.

A container watches the league on the §9.1 cadence, recomputes rest-of-season
value and Δ P(title) with the M2-M7 engines, and EMAILS an ordered list of
actions. §0.1 is absolute here and asserted by test: nothing in this package
writes to ESPN, because the one place an unattended agent could do real damage
is the one place nobody is watching.
"""
