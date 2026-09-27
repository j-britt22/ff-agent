"""Delivery. Email is the only backend shipped; the interface is what matters.

§6.3: email is right for the Tuesday waiver list and the Monday trade report and
is a WEAK channel for an 11:15 Sunday alert, because it buzzes a phone only if a
rule says it should. Stated plainly rather than papered over. The notifier stays
behind an interface so a push backend is about thirty lines and touches nothing
else — and the Sunday job logs its own send latency precisely so whether one is
needed becomes measurable rather than a matter of opinion.
"""

from ff_agent.inseason.notify.base import (
    Digest, MemoryNotifier, Notifier, NullNotifier, SendResult, SecretLeak,
    assert_no_secrets,
)

__all__ = [
    "Digest", "MemoryNotifier", "Notifier", "NullNotifier", "SendResult",
    "SecretLeak", "assert_no_secrets",
]
