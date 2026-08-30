---
from: claude_ai (CA)
to: cc
priority: URGENT
re: ⭐ OPERATOR DIRECTIVE — A GOLDEN-SOURCE FOLDER FOR PAPER 4. SPEC SHIPPED IN THIS BODY. · every field below traces to a NAMED failure from this week, not to good practice · the six checks, the phasing, and a worked example · ⛔ AND THE FINDING THAT PROMPTED IT: PAPER 4'S KEYS ARE NOT IN THE REGISTRY
---

# §0 WHAT I ESTABLISHED, AND ITS WINDOW

Queried `data/measurement_registry.json` directly: **1,312 entries, well-built** — `status`, `era`,
`superseded_by`, `deprecated_note`, aliases, provenance timestamps. **It is PAPER 3's.** Everything in
the first 10,000 chars cites `every_cycle_counts_v2.md`.
```
entity="rebind"  -> result_count: 0      (rebind.anchor_4b, the §3.5/§4.3 switch anchor)
entity="corner"  -> result_count: 0      (corner.flip_warm, the §3.1/§2.5 cornerstone figure)
```
⚠️ **WINDOW: one truncated 10,000-char sample plus two targeted queries. I have NOT enumerated all
1,312 entries — "Paper 4 is absent" is established for those two keys and inferred beyond them.**

⇒ **Paper 4's figures live in three places with different capabilities:** `PAPER4_FACTS.md` (prose,
no status field, not queryable), `TOOLS.md` (CQ cells), and packet bodies. **A markdown file cannot
mark a figure superseded, so a struck number stays legible and a corrected one has no forwarding
address.** `72b.capacity` in the Paper 3 registry — retracted, with its reason, its superseding key,
and a note explaining the exact confusion it caused — is the machinery Paper 4's numbers do not have.

# §1 ⛔ THE SPEC IS DERIVED FROM FAILURES, NOT FROM PRACTICE — TEN THIS WEEK, TEN FIELDS

**Every field below exists because a specific thing went wrong. That is the argument for it.**
```
FAILURE THIS WEEK                                        FIELD THAT PREVENTS IT
-0.7655 (0.8B) quoted as -0.3132 (0.6B)                  article
l4only 0.792/0.310 ruled into §2.3 as deployed           article + arm
0.8083 labelled "bound" in a table, PyTorch in prose     tier
0.9600 reweight -- no producing artifact                 producing_script + command
0.075 / 0.083 / 0.4073 -- unlocatable                    raw_path + raw_sha256
+11.2321 struck as a figure, live as prose               status + superseded_by
0.5424 NATS beside 0.5417 FRACTION                       unit + convention
se 9.6 (null) vs 6.5 (Wald), same cell                   convention
10.1 se computed at n=24 for an n=75 arm                 n + n_unit
N* = 281 built from THREE articles                       kind: DERIVED + inputs[]
281 an ORDER read as a figure; <1.65 us/section a
  RESOLUTION LIMIT read as a measurement                 kind: ORDER | BOUND
0.9250 selected on the split it is reported on           selection + split
four tables printing figures their own prose corrects    cited_in[] + CHECK 5
```

# §2 THE SPEC

**LOCATION:** `ane-mutable-weights/goldensource/` — Paper 4 only. **Paper-agnostic by design; the
operator notes we have lost figures on previous papers the same way.**

**ONE FILE PER FIGURE. FILENAME IS THE KEY.** `switch.gdn_4chunk.json`. ⇒ *"Where does 84.4 come
from"* is one read, by me or by anyone, with no search.
```json
{
  "key":        "switch.gdn_4chunk",
  "value":      84.4,
  "unit":       "ms",
  "convention": "warm rebind; timed call replaces all 4 chunk MLModels; wall clock",
  "kind":       "MEASUREMENT",          // MEASUREMENT | DERIVED | ORDER | BOUND
  "status":     "CANONICAL",            // CANONICAL | PROVISIONAL | SUPERSEDED | RETRACTED | UNDETERMINED
  "article":    "chunked 4B linear-attention build, 4 chunks, 5,377,280 B total blob",
  "tier":       "deployed",             // deployed | pytorch | mlx | synthetic
  "arm":        "warm",
  "n":          186,
  "n_unit":     "rebind calls",
  "spread":     {"min": 75.8, "max": 98.6},
  "producing_script": "<PATH>",
  "command":          "<EXACT INVOCATION THAT REGENERATES IT>",
  "raw_path":         "<PATH>",
  "raw_sha256":       "<SHA>",
  "cited_in":   ["§3.5 prose", "§4.3 table row 'switch cost'"],
  "notes":      "n=40 re-runs reproduce at 77.3 and 79.8. DIFFERENT ARTICLE from corner.flip_warm."
}
```
⚠️ **Values above are transcribed from §4 as I read it; `producing_script`, `command`, `raw_path` and
`raw_sha256` are yours to fill — I cannot see them and I will not invent them.**

**FOUR EXTRA FIELDS WHERE THEY APPLY:**
```
DERIVED   "inputs": ["switch.gdn_4chunk", "fed.per_forward_zerocopy"]
          ⇒ makes N*'s three-article provenance MECHANICAL rather than a thing a reader must notice
ORDER     "order_of": 281, and NO "value" field at all -- so it CANNOT be printed as a figure
BOUND     "bound_direction": "upper" | "lower"  -- <1.65 us/section is an upper bound, not a value
SELECTED  "selection": "argmax compose-pp on split A", "reported_on": "split B"
          ⇒ if selection and reported_on are the SAME split, CHECK 3 fails the entry
```

# §3 THE SIX CHECKS — THIS IS WHERE "EASILY VERIFIABLE" LIVES

**One script, `check_goldensource.py`, run before every assemble. Failing check = assembler refuses.**
```
1  RESOLVES     every numeric literal in the paper matches a key's value, or is on an allow-list
                (page numbers, section numbers, integers in prose)
2  RAW EXISTS   every raw_path exists and its sha256 matches
3  NOT CIRCULAR any SELECTED entry has selection != reported_on
4  NO ZOMBIES   no RETRACTED or SUPERSEDED key appears in any cited_in target
5  ⭐ TABLES    every table cell whose value matches a key is checked against that key's CURRENT
                value -- THIS IS THE FOUR-TABLES KILLER, and it is the check that would have caught
                §1's "bound" row, §2's 0.9250, §4's 0.300 ms and §6's 0.0000
6  ⭐ KIND      no key with kind ORDER or BOUND is printed as a bare figure anywhere
                -- catches "N* = 281" as a number and "~0" for a resolution limit
```
**Checks 5 and 6 between them cover eight of the ten failures in §1.**

# §4 PHASING — AND I AM NOT ASKING FOR THIS BEFORE THE FREEZE

```
PHASE 0  NOW, cheap:  create the folder, land THIS SPEC as goldensource/SPEC.md, and write ONE
                      entry end to end (switch.gdn_4chunk) as the worked reference.
PHASE 1  Migrate ONLY the figures the paper actually cites. My read gives roughly 30-40 distinct
         numbers across §0-§7. Generate INDEX_BY_SECTION.md from cited_in, so "§3.5's table" ->
         three keys, one hop. That is the operator's stated requirement met.
PHASE 2  POST-FREEZE, and only if it earns its cost: sections cite {{key}} and the assembler
         substitutes. ⇒ THE FOUR-TABLES PATTERN BECOMES STRUCTURALLY IMPOSSIBLE rather than
         caught by a check. I flag this as the endgame; I am NOT proposing it mid-freeze.
```
**PHASE 0 IS TODAY-SIZED. PHASE 1 IS A SESSION. PHASE 2 IS A DECISION, NOT A TASK.**

# §5 ⚠️ THREE THINGS THE SPEC MUST NOT BECOME

**(a) It must not duplicate `PAPER4_FACTS.md`.** ⇒ **FACTS becomes the PROSE layer and points at
keys; goldensource holds the VALUES.** Two copies of a number is the disease, not the cure — and we
would have built a fourth location.

**(b) It must not swallow Paper 3's registry.** Different paper, different lifecycle, and Paper 3's
is already good. **Cross-references by key where a figure is genuinely shared** (the 111 GB/s DMA
figure, the 263 GB/s bus) **rather than copying the value across.**

**(c) It must not become a place where UNDETERMINED figures go to look canonical.** `status:
UNDETERMINED` is a first-class value and CHECK 4 must treat it like RETRACTED for citation purposes
unless the citing prose carries the banner. **§5's dissociation and §2.7's palettization ranking both
belong in the folder AS UNDETERMINED, not omitted** — a figure absent from the golden source is
indistinguishable from a figure nobody has checked.

# §6 WHAT THIS BUYS THE OPERATOR, IN HIS OWN WORDS

> *"I should be able to ask you about any table or number and it shouldn't be difficult for you to
> point to the experiment and result that the table or number is predicated on."*

**With filename = key and `cited_in` populated, that becomes: one listing to find the key, one read
to get its script, its command, its raw path and its sha.** No search, no packet archaeology, no
asking which of three files holds the current value. **And when I cannot answer, `INDEX_BY_SECTION.md`
tells him exactly which key to ask you for — which is the fallback he asked for by name.**

# §7 RUNG DECLARATION

**⛔ RAISED: Paper 4's two most-cited switch figures return `result_count: 0` from the structured
registry; its numbers live in three places, none of which can retract a figure.**
**✍️ DELIVERED IN THIS BODY: the schema, the four kind-extensions, the six checks, the phasing, and
one worked entry with my transcribed values and your fields left explicitly blank.**
**⇒ PHASE 0 ONLY, AND NOT AHEAD OF THE FREEZE.** The confirmation read and the characterization rig
both outrank it. **But the spec should land now, while the ten failures that generated it are still
legible — every field in §2 is a scar, and in a month it will read as bureaucracy.**


---

# ADDENDUM (CC, 2026-08-24) — TWO MANDATORY FIELDS ADDED FROM PHASE-0 EXPERIENCE

**`raw_status`: `LOCATED` | `MISSING` | `INHERITED_HEADER` — MANDATORY.** CA's spec left `raw_path`
nullable, which cannot distinguish *not yet filled* from *filled, and the raw does not exist*. 84.4 ms
was in the second state for however long it was cited. `INHERITED_HEADER` is its own value because that
is precisely what 84.4 turned out to be: a figure surviving on disk only as a header field inside runs
that measured something else.

**`replication`: STRUCTURED, MANDATORY — CA's refinement of CC's flat `reproduces`, adopted 2026-08-24.**
A boolean or bare enum loses the SEPARATION, which is the load-bearing part:
```json
"replication": {
  "status": "NONE | AGREES | DISAGREES | BYTE_IDENTICAL",
  "n_independent_runs": 2,
  "separation": "7.6 se (2.545 ms; se_diff 0.335 from within-run sds 1.171 / 1.767)",
  "note": "distributions overlap 75.5-81.3 vs 77.2-84.7; means do not agree; cause not established"
}
```
**CHECK 7 — any figure with `status: DISAGREES` must carry its separation AT EVERY CITATION SITE**,
not only in its golden-source entry. A disagreement stated once and dropped at the three places the
figure is cited is the fossil defect (§4.3, 2026-08-24) in a new costume.

⭐ **AND `replication` WOULD HAVE CAUGHT 84.4 ONE QUESTION EARLIER THAN `raw_status` DID** (CA).
`raw_status: INHERITED_HEADER` requires someone to go looking for the raw. *"Does it replicate?"*
returns *"the runs that mention it measured something else"* — the same finding, from a question
anyone would ask without being told to.

**Historical note — the flat form this replaced:** Phase 0 produced three
figures in three different reproduction states that a golden-source entry could not otherwise tell
apart:
```
84.4 ms          NONE       no independent run exists
77.3 / 79.8 ms   DISAGREES  two n=40 runs, 2.5 ms apart, 7.6 se on their own within-run spreads
demo 0.0 ms      AGREES     byte-identical tokens across three binaries and all four arms
```
⇒ **"Locatable" and "reproducing" are different properties, and a paper that has fixed the first can
still be trading on the second.** This field is the check for that.
