# Blind Style-T Multicandidate Rater v1

Rate every anonymous Chinese candidate against the supplied author-anonymous
Style T definition. The English source is semantic authority, not a writing
template. Ignore topic, genre, names, lore, and guesses about authors, models,
or methods.

For each candidate:

1. Rate repeated structural Style T adherence from 1 to 5, based on syntax,
   rhythm, discourse flow, dialogue realization, and source-licensed
   punctuation.
2. Rate Chinese naturalness from 1 to 5.
3. Give one concise, text-grounded rationale.

Use the full 1-5 range when warranted. Do not reward semantic invention or
copying merely because it resembles the definition. Return one judgment for
every candidate ID in input order and only schema-conforming JSON.
