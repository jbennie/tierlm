# Fragments: distilled designs from eight MoE inference engines

This folder is the design input for tierlm. Each **fragment** is a short
pseudocode description of one algorithm, data layout or measured lesson, written
in our own words after reading one or more reference engines. **CONSENSUS.md**
groups the fragments into patterns, counts how many independent projects
converge on each, and gives a verdict. The library is built from the patterns
with the most agreement, and the measured failures are kept so they are not
repeated.

## Files

| File | What it holds | Fragments |
|---|---|---|
| FORMAT.md | The fragment template and the no-copy rule | - |
| CONSENSUS.md | Patterns per topic with origins, counts, verdicts, contradictions, negative-results ledger | - |
| storage_cache_io.md | Expert storage on NVMe, RAM expert cache, read paths, prefetch (`F-store`, `F-cache`, `F-io`, `F-pf`) | 25 |
| quant_kernels.md | Block formats, dequant maths, CPU dot kernels, dispatch (`F-quant`) | 23 |
| forward_pass.md | Transformer/MoE numerics: norms, RoPE, attention, routers, MTP, KV, sampling, oracle (`F-fwd`) | 30 |
| support.md | Tokenizer, file readers, hardware probing and planning, config, HTTP server, JSON (`F-TOK`, `F-FMT`, `F-HW`, `F-CFG`, `F-SRV`, `F-JSON`) | 25 |
| origin_apus.md | Fragments from apus alone (GLM-5.3-Flash C11 engine) | 25 |
| origin_flashmoe.md | Fragments from flash-moe alone (SSD-streamed 397B MoE on a laptop), plus its experiment ledger | 24 |

The four multi-project files end with a "Recommended synthesis"; the two
single-project files end with "Notable ideas" / "Experiment ledger". Those
sections predate the consensus scoring; where they disagree with CONSENSUS.md,
CONSENSUS.md wins.

### Reference projects

| Tag | Project | Licence |
|---|---|---|
| colibri | colibri | Apache-2.0 |
| ds4 | ds4 (DwarfStar) | MIT |
| pulsar | pulsar | MIT |
| monolite | monolite | Apache-2.0 |
| slipstream | slipstream (new code under engine/ only; the rest is llama.cpp) | MIT |
| ktransformers | ktransformers kt-kernel | Apache-2.0 |
| apus | apus-glm5.3-flash | PolyForm-NC-1.0.0 |
| flashmoe | flash-moe | no licence file |

Not all of them are independent: apus adapts colibri's expert store, cache,
memory guard, prefetch pilot, compat shims and gateway; slipstream takes its
disk-benchmark method, coupling analysis and RAM-admission idea from colibri;
the GGUF block formats come from GGML. CONSENSUS.md marks these with †.

## How to read a fragment

```
## F-cache-08: Route-hotness LFU with periodic halving and hit protection
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4_metal.m ...; pulsar (MIT), ... -- read only, not copied
- Purpose / Inputs / outputs / Pseudocode / Notes / Variants seen elsewhere
```

- **Id.** `F-<topic>-<nn>`. Ids in the four multi-project files are unique. The
  single-project files reuse short ids (`F-io-01` exists in both), so
  CONSENSUS.md writes them `apus/F-...` and `fmoe/F-...`.
- **Origin** lists every project where the fragment's pattern appears, which
  can be more than the projects named in `Source` (other projects found through
  "Variants seen elsewhere", other fragments of the same pattern, or a grep of
  the reference trees). It can also be fewer: a variant that is a different
  pattern is not counted (for example monolite's history-ranked lookahead is not
  counted for the router-lookahead fragment `F-pf-21`).
- **Licence-of-origin** groups the Origin tags by licence. It is provenance, not
  permission to copy anything.
- **Source** is where the fragment was read; **Variants seen elsewhere** is how
  other projects differ.
- **Numbers** are each project's own measurements on its own hardware, quoted so
  the claim can be checked, not our results.
- To see how much a fragment's pattern is trusted, find its id in the Fragments
  column of CONSENSUS.md and read the Count and Verdict there.

## The no-copy rule

Fragments carry **ideas, never code**. When writing or editing a fragment:

- Read the reference source only to understand it; then write pseudocode from
  scratch, with our own names and structure.
- Do not copy source lines, comments, identifiers (beyond standard terms such as
  `top_k` or `rope`), regexes, lookup tables or test data. Point to the public
  specification instead (for example the GGML/GGUF tables).
- Search local clones of the reference projects (kept outside this repo) with grep to confirm
  whether a pattern exists there; do not paste what you find.
- Patterns whose only origins are apus (non-commercial licence) or flash-moe (no
  licence) are re-derived from first principles or public specs and reviewed
  with that in mind.

## Adding a fragment

1. Pick the file by topic (a single-project file is only for material from one
   project) and the next free number for its prefix.
2. Write it with the FORMAT.md template, `Source` first; pseudocode, notes and
   variants in your own words.
3. Find the pattern it belongs to in CONSENSUS.md:
   - If it describes an existing pattern, add its id to that row's Fragments
     column (before `also:` if the fragment is mainly about the pattern, after
     it if it only mentions or measures it).
   - Otherwise add a row with the next free `P-` number at the end of the topic
     table; the numbers are ids, not ranks.
4. Grep the reference trees for the pattern in the other projects and add any
   confirmed tags to the row's Origins.
5. Re-score the row (below), then give the fragment its `Origin:` line (the
   union of the Origins of every row that lists it before `also:`) and its
   `Licence-of-origin:` line.
6. If it records a failed experiment, add a row to the negative-results ledger;
   if it records a bug that produced plausible output, add it to the
   silent-correctness traps.

## Re-scoring consensus

Re-score a row whenever its Origins change: a new fragment, a new grep finding,
or a new measurement.

1. **Origins.** A project counts when it implements the pattern in its own code,
   or ran and measured it. Write the tags in the fixed order colibri, ds4,
   pulsar, monolite, slipstream, ktransformers, apus, flashmoe.
2. **Count** = number of tags.
3. **Verdict:**
   - `reject` if any project measured the pattern negative (this overrides the
     count; record the numbers in Notes and in the ledger);
   - otherwise `adopt` if Count >= 3, or if it is the only correct way
     (model-defined maths, a file-format definition, or a correctness
     requirement; say which in the cell);
   - `adopt-tested` if Count = 2;
   - `experimental` if Count = 1.
4. **Lineage.** If the tags include a linked pair (apus with colibri on the
   apus-adapted mechanisms; slipstream with colibri on the colibri-derived
   ones), start Notes with `†` and give the independent count. Treat a †
   `adopt` whose independent count is below 3 as weaker evidence.
5. **Contradictions.** If projects disagree and there are measurements on both
   sides, add or update an entry in the Contradictions section and put the
   numbers for both sides in Notes.
6. **Propagate.** Rewrite the `Origin:` and `Licence-of-origin:` lines of every
   fragment listed before `also:` in that row, and update the Totals paragraph.

Quick checks:

```sh
# every fragment has exactly one Origin and one Licence line
for f in storage_cache_io.md quant_kernels.md forward_pass.md support.md origin_apus.md origin_flashmoe.md; do
  echo "$f $(grep -c '^## F-' $f) $(grep -c '^Origin: ' $f) $(grep -c '^Licence-of-origin: ' $f)"
done

# Origin lines use only the eight tags
grep -h '^Origin: ' *.md | grep -vE '^Origin: (\[(colibri|ds4|pulsar|monolite|slipstream|ktransformers|apus|flashmoe)\] ?)+$'

# fragments a given project contributes to
grep -B1 '^Origin: .*\[flashmoe\]' *.md | grep '^.*## F-'

# verdict tally
grep -oE '\| (adopt|adopt-tested|experimental|reject)( \([^|]*\))? \|' CONSENSUS.md | cut -d' ' -f2 | sort | uniq -c
```
