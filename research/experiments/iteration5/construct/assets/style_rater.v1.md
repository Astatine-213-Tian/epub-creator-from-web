# Blind Style-T Pair Rater v1

For each pair, compare anonymous Chinese candidates A and B against the supplied
author-anonymous Style T definition. The English source fixes content but is not
a writing template. Ignore topic, genre, names, lore, and whether a candidate
seems machine-written. Do not infer an author, method, or hidden candidate role.

For every pair:

1. Choose `A`, `B`, or `tie` for stronger repeated structural adherence to
   Style T while preserving the English meaning.
2. Rate each candidate's Style T adherence from 1 to 5.
3. Rate each candidate's Chinese naturalness from 1 to 5.
4. Give one concise sentence grounded in observable syntax, rhythm, discourse,
   dialogue, or punctuation behavior. Do not name an author or method.

Return exactly one judgment for every pair ID in input order and only the
schema-conforming JSON.
