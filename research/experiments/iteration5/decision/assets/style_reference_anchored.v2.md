# Blind Reference-Anchored Structural Style Rater v2

The `style_reference` is an evaluation-only clean original-Chinese passage.
Rate each anonymous candidate for passage-specific structural similarity to
that reference. This task does not identify an author and does not evaluate
semantic fidelity to an English source.

Use only these dimensions:

1. `clause_packaging_and_syntax`: clause chaining, subordination, constituent
   ordering, and subject/object realization.
2. `sentence_rhythm`: alternation of short and long sentences, cadence, and
   sentence-boundary placement.
3. `paragraph_discourse_progression`: how actions, perceptions, reactions, and
   transitions are staged across paragraphs.
4. `dialogue_turn_and_attribution_architecture`: turn segmentation, placement
   and form of speech attribution, and integration of action with dialogue.
5. `narrator_stance_and_reaction_timing`: narrator distance, focalization, and
   when internal or external reactions enter the sequence.
6. `punctuation_and_emphasis`: punctuation inventory, pause structure,
   repetition, ellipsis, and marked emphasis.
7. `function_word_and_register_texture`: function-character usage, connective
   texture, and level of written/colloquial register without rewarding shared
   content words.

For a dimension that is not observable in the reference passage, return
`null`; do not infer it from the candidate. Ignore names, lore, topic, factual
differences, and guesses about authors, models, or methods. Do not reward
word-for-word copying, shared content words, or matching proper nouns.

For every candidate:

1. Score every applicable dimension from 1 to 5 and use `null` for N/A.
2. Rate overall passage-specific structural similarity from 1 to 5, based only
   on applicable dimensions.
3. Rate Chinese naturalness from 1 to 5.
4. Give one concise rationale grounded in observable structural evidence.

Return one judgment for every candidate ID in input order and only
schema-conforming JSON.
