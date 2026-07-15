# English Semantic Adjudication v1

Independently compare the Chinese source with the draft English, then return a
corrected plain-English semantic source. This is benchmark construction, not
literary translation.

Rules:

1. Preserve every event, relation, entity, number, speaker, speech act,
   chronology, cause, negation, modality, and intensity.
2. Correct every omission, invention, role reversal, or ambiguity in the draft.
3. Keep the English plain and natural; do not preserve Chinese literary style.
4. Keep every paragraph ID and order exactly once.
5. Do not summarize, censor, embellish, explain, or add notes.
6. Set `approved` true only when the returned English is semantically faithful.

The request JSON follows this prompt. Return only the schema-conforming JSON.
