"""M10 — post-draft analysis: what the draft actually did, and what it implies.

Three questions, deliberately kept apart because they are known to different
precisions:

* **Every pick, graded** (``grade.py``) — the strongest of the three. It scores a
  pick against the board that was live at that moment and against the league's
  own opponent model, both of which are measured artefacts.
* **Team ratings** (``teams.py``) — solid on draft SHAPE (who has starters, who
  has holes, who is exposed to byes), weak on fine distinctions between rosters
  of similar projected value. That limit is M6's, not this module's.
* **Final standings and playoff odds** (``predict.py``) — the WEAKEST, and it
  ships with the number that says so. M6 measured this exact task on 2025 and
  scored Spearman -0.21; a PERFECT simulator scores only +0.52 on it. Anyone
  reading a projected finishing order as a forecast is reading it wrong.
"""
