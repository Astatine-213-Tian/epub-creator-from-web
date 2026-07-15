# Blind English-Grounded Multicandidate Validator v1

Evaluate every anonymous Chinese candidate against the supplied English source.
Do not judge author identity or reward stylistic similarity.

For each candidate:

1. Rate semantic fidelity from 1 to 5.
2. Rate Chinese naturalness/readability from 1 to 5.
3. Mark whether any high-severity semantic error is present: invented or
   missing event, reversed relation, altered negation/modality/intensity,
   wrong speaker, materially changed entity, or broken causality/order.
4. Mark whether speaker assignments, dialogue turns, and paragraph topology
   are preserved.
5. Give one concise rationale naming the most important evidence.

Return one judgment for every candidate ID in input order and only
schema-conforming JSON.
