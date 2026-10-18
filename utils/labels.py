"""Canonical disease names in the original network output-index order.

Output indices are not the same as the display order used in manuscript tables.
Preserve these indices for checkpoints trained on the original four-class labels.
"""
CLASS_NAMES = ("FTA", "WI-FTC", "EA-FTC", "MI-FTC")
PAPER_CLASS_ORDER = (0, 3, 2, 1)
PAPER_CLASS_NAMES = tuple(CLASS_NAMES[i] for i in PAPER_CLASS_ORDER)
PROBABILITY_COLUMNS = ("P_FTA", "P_MI_FTC", "P_EA_FTC", "P_WI_FTC")
