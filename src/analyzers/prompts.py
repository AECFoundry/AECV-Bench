"""
Prompts for LLM floor plan analysis.
"""

# Counting rules and interpretation guidelines
COUNTING_RULES = """
1. Counting Rules and Interpretation Guidelines

You must follow these rules precisely—these match the logic used in preparing the ground-truth dataset. All counts MUST be based on the entire floor-plan image.

1.1 General

- Always consider the entire drawing before counting anything.
- Understand the entire drawing as a architectural and/or engineering drawing before counting anything.
Drawings may use different languages, symbols, and conventions; interpret them consistently.

1.2 Doors

- Count only openable doors (swing doors drawn with an arc).
- Do NOT count sliding closet doors, pantry sliders, or furniture-like wooden sliders on internal walls.
- Each leaf counts separately, i.e, a double-leaf door = 2 doors and a triple-leaf door = 3 doors, etc. and so on.
- Garage entrances are NOT doors.

1.3 Windows

- All sliding openings are counted as windows.
- French doors or large glazed openings are counted as windows, NOT doors.
- Multiple adjacent windows (or sliding units) without a gap between them count as ONE single window group.
- Tiny toilet windows must be counted even if very small.
- Garage entrances are NOT windows. 

1.4 Spaces / Rooms

- Count all enclosed spaces, even if unlabeled.
- Partnerships of adjacent but separated spaces (e.g., WC next to Bath) are counted individually.

1.5 Bedrooms

A space labeled "Bedroom,","Chambre," "Zimmer," "Camera," etc., or equivalent should be counted as a bedroom.

1.6 Toilets

- The following labels all count as toilet units, each counted individually:
WC, W.C., Bath, Bain, Douche, Shower, SDB, Bathroom, Toilet, etc.
- Lavatory are not counted as toilets.

Make sure your counts follow all the rules above.
"""