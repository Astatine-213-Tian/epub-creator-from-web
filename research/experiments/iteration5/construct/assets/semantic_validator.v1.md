# Blind English-Grounded Semantic Validator v1

Evaluate each anonymous Chinese candidate only against its English semantic
source. Do not infer an author, method, or hidden candidate role. Do not compare
items with one another.

For every item:

1. Rate semantic fidelity from 1 to 5. A 4 or 5 requires preservation of all
   material events, states, participants, relations, chronology, causality,
   negation, modality, intensity, and speaker attribution.
2. Rate Chinese naturalness from 1 to 5.
3. Mark a high-severity semantic error only for a changed speaker/action/
   relation, reversed polarity or modality, invented consequential event, or
   material omission.
4. Mark whether dialogue turns and speaker topology are preserved.
5. Give one concise evidence-based sentence. Do not name an author or method.

Return exactly one judgment for every item ID in input order and only the
schema-conforming JSON.
