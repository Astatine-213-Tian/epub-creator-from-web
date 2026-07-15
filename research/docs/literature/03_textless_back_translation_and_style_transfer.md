# Note 03: Textless Back Translation and Style Transfer

## Useful Insight

The workflow is not ordinary translation. It is closer to textless back translation:
the English is the semantic source, while the Chinese corpus supplies a target-language
stylistic root. The system should say "plausible Chinese reconstruction", not "recovered
original Chinese".

## Source Notes

### Wang 2024: textless back translation reconstructs invisible source texture

Source: Binhua Wang, "The creation of new academic knowledge spaces through the
repatriated self-translation of foreign-language texts", accepted manuscript in
Target. https://eprints.whiterose.ac.uk/id/eprint/221869/

Relevant finding:

- The paper discusses "rootless" / "textless" back translation for foreign-language
  texts with Chinese cultural roots but no ordinary Chinese source text.
- It describes the need to reconstruct an invisible Chinese source-like layer when
  a foreign-language text contains implicit Chinese cultural material.

Pipeline implication:

- Eternal Gate is a narrower, author-style variant of this problem. The hidden source
  is not available, but the author's Chinese writing corpus provides style roots.
- The pipeline should keep an epistemic boundary:
  "recover meaning from English" + "constrain Chinese style from corpus", not
  "restore the original wording."

### Textless back translation standards: language, style, culture

Source: "A Study of Textless Back Translation from the Perspective of Intertextuality",
Atlantis Press 2021.
https://www.atlantis-press.com/proceedings/ichssr-21/125956866

Relevant finding:

- The abstract frames textless back translation around language, style, and cultural
  standards.

Pipeline implication:

- Keep three separate constraint classes:
  - language standard: fluent Simplified Chinese, punctuation, paragraph boundaries;
  - style standard: author-like cadence and scene mechanics;
  - cultural/worldbuilding standard: names, religious terms, mythic logic, comments,
    glossary, and source continuity.

### Prabhumoye et al. 2018: style transfer means changing style while preserving intent

Source: "Style Transfer Through Back-Translation", ACL 2018.
https://aclanthology.org/P18-1080/

Relevant finding:

- Text style transfer is framed as rephrasing text to carry specific stylistic
  properties without changing intent or affect.
- Back-translation is used to learn meaning-grounded representations and reduce
  source-style artifacts.

Pipeline implication:

- Our problem is not model training, but the principle still applies: separate
  semantic content from style pressure.
- A robust flow should either:
  - generate a semantic draft first and then style-revise under strict fidelity gates,
    or
  - generate once with a compact style constraint and validate style/content separately.

### Rabinovich et al. 2017: author traits can vanish through translation

Source: "Personalized Machine Translation: Preserving Original Author Traits", EACL
2017. https://aclanthology.org/E17-1101/

Relevant finding:

- The paper studies author traits in original texts and translations.
- It reports that authorial signals can be obscured in both human and machine
  translation, and proposes adaptation methods to retain traits.

Pipeline implication:

- If the English source is already a translation or publication layer, it may have
  lost authorial Chinese signals.
- The Chinese style layer should not be a decorative polish after the fact; it needs
  to actively reintroduce target-language authorial mechanics where semantically
  licensed.

### Jin et al. 2022, Reif et al. 2022, CAT-LLM 2024/2025: LLM style transfer needs explicit style definition

Sources:

- Di Jin et al., "Deep Learning for Text Style Transfer: A Survey", Computational
  Linguistics 2022. https://aclanthology.org/2022.cl-1.6/
- Emily Reif et al., "A Recipe for Arbitrary Text Style Transfer with Large Language
  Models", ACL 2022. https://aclanthology.org/2022.acl-short.94/
- Zhen Tao et al., "CAT-LLM: Style-enhanced Large Language Models with Text Style
  Definition for Chinese Article-style Transfer", arXiv 2024/2025.
  https://arxiv.org/abs/2401.05707

Relevant finding:

- TST research commonly separates content preservation, style transfer, and fluency.
- Prompted LLM style transfer can work, but style descriptions must be operational.
- CAT-LLM is especially relevant because it builds a pluggable style-definition
  module for Chinese long-text style transfer.

Pipeline implication:

- Replace a long Chinese "style guide" with a structured Text Style Definition:
  global signals, scene evidence cards, genre/domain weights, negative GPT-style
  patterns, and evaluation hooks.
- The model should receive compact constraints; the full evidence stays in artifacts
  for audit and retrieval.

## Design Rule

Use a two-stage conceptual contract:

1. **Semantic contract**: preserve English meaning, names, comments, sequence, and
   paragraph boundaries.
2. **Style contract**: choose Chinese forms that are supported by the author's corpus
   and appropriate for the paragraph's scene/domain.

If the two conflict, semantic fidelity wins and the validator should report that the
style opportunity was skipped.
