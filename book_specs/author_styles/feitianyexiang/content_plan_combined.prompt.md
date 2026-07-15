# Style-Transfer Full-Regeneration Prompt v1

## Task

Regenerate every Chinese paragraph using the supplied frozen method payload.
The English semantic source is authoritative. The neutral Chinese is a
terminology and content-preservation anchor, not a syntax template. Apply only
target-style tendencies licensed by both the method evidence and the current
English paragraph.

## Non-negotiable constraints

1. Preserve every paragraph ID and its order. Return one output paragraph for
   every input paragraph.
2. Preserve facts, events, event order, causality, negation, modality,
   intensity, quantities, Latin tokens, entities, and speaker attribution.
3. Preserve whether each paragraph contains dialogue, its speaker sequence, and
   whether it begins with direct dialogue. Do not invent dialogue, gestures,
   thoughts, emotions, imagery, lore, or transitions.
4. Regenerate the complete paragraph. Do not merely patch punctuation or swap a
   few words, but do not expand, summarize, or embellish the content.
5. Reference passages teach only abstract rhythm, syntax, discourse, dialogue,
   and punctuation patterns. Never copy eight or more consecutive Chinese
   characters or import their wording, events, entities, or imagery.
6. Corpus rates are tendencies, not quotas. Skip a cue when its semantic trigger
   is absent.
7. For `content_plan_combined_full_regeneration`, first derive a compact
   paragraph-level content plan from English and return it. For every other
   method, return an empty `content_plan` array.
8. Return valid JSON only. Paragraph text must not contain JSON wrappers,
   Markdown fences, commentary, labels, or analysis.

## Input

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "strong",
  "english_semantic_source": [{"id": "p0001", "en": "..."}],
  "neutral_zh": [{"id": "p0001", "zh": "..."}],
  "method_payload": {},
  "reference_examples": []
}
```

## Output

Return JSON only:

```json
{
  "sample_id": "...",
  "method_id": "...",
  "intensity": "strong",
  "content_plan": [
    {"id": "p0001", "facts": ["..."], "constraints": ["..."]}
  ],
  "paragraphs": [{"id": "p0001", "zh": "..."}],
  "style_cues_applied": [],
  "style_cues_skipped": [],
  "uncertainties": []
}
```
