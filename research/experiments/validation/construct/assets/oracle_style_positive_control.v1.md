# Target-Style Positive-Control Reconstruction v1

Generate a high-access positive control for validating a style meter. The
English paragraphs are the semantic authority. The neutral Chinese is a content
and terminology anchor, not a sentence template. The supplied evidence defines
anonymous target Style T.

Rules:

1. Rewrite every paragraph into natural Chinese that consistently realizes
   Style T when the source supports it.
2. Preserve all events, entities, numbers, speakers, dialogue topology,
   chronology, causality, negation, modality, and intensity.
3. Do not add dialogue, imagery, gesture, emotion, intimacy, lore, or motives.
4. Keep every paragraph ID and order exactly once.
5. Use structural tendencies rather than topic, names, genre, or remembered
   passages as evidence of style.
6. Never copy eight consecutive CJK characters from a reference example.
7. If a style tendency conflicts with meaning, preserve meaning.

This is an oracle-access construct control, not a production candidate and not
an efficacy result for a transfer method. The request JSON follows this prompt.
Return only the schema-conforming JSON.
