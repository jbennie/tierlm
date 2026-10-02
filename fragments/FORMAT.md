# Fragment format

Each fragment is pseudocode written after reading an existing open-source
implementation. Fragments contain NO copied source code, NO copied comments and
NO copied identifiers beyond standard technical terms (e.g. "top_k", "rope").
They describe the algorithm, data layout and decisions in our own words.

Template:

## F-<topic>-<nn>: <short title>
Origin: [<tag>] [<tag>] ...

Licence-of-origin: <licence> (<tag>, <tag>), <licence> (<tag>)
- Source: <project> (<licence>), <path(s)> -- read only, not copied
- Purpose: one line
- Inputs / outputs: types and shapes
- Pseudocode:
      (indented, language-neutral, our own naming)
- Notes: numerics, edge cases, why this variant, measured claims from the project
- Variants seen elsewhere: <project>: how it differs

`Origin:` lists, in the order colibri, ds4, pulsar, monolite, slipstream,
ktransformers, apus, flashmoe, every project where the fragment's pattern
appears (not only the one read for `Source:`). `Licence-of-origin:` groups those
tags by licence. Both lines are kept in sync with CONSENSUS.md; see README.md.
