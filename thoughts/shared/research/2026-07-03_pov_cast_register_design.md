---
date: 2026-07-03
owner: research (scout)
status: design proposal — NON-training, not implemented
topic: Character cast register to resolve gendered-pronoun / POV failures by explicit conditioning
supersedes-note: >
  The referenced audit doc (thoughts/shared/research/2026-07-02_pipeline-audit-synthesis.md §4.3)
  is NOT present in this worktree — thoughts/shared/research/ was empty before this file.
  Findings below are re-derived directly from source (file:line cited), not from that doc.
---

# POV / gendered-pronoun fix via a character cast register (design, non-training)

## TL;DR verdict

**Feasible, medium build cost, low serve risk — recommended as the next non-training POV lever, ranked just under image-as-context.** The prompt-injection seam already exists, is already proven byte-safe by a golden test, and the batch pipeline already holds a persistent per-chapter object that can accumulate a rolling register for free. The hard part is *building* an accurate register, not injecting it. The browser single-page path degrades to today's behaviour (static/empty cast) with no regression.

The core bet: **JP pro-drop means one bubble underdetermines gender; the cast identity lives at chapter scope. Move gender from per-bubble *inference* (what training tried, and hit a ceiling on) to explicit *conditioning* (a name→gender/role table in the prompt).** This is the standard "entity/glossary conditioning" trick from MT, applied to pronouns instead of terminology.

---

## 1. What exists today (verified, file:line)

There are **two disconnected, single-title, hardcoded mechanisms**. One touches the prompt, one touches the output. Neither is enabled in prod.

### 1a. `DEFAULT_CAST_ANCHOR` — prompt-side, DISABLED (`settings.translation_cast_anchor=False`)

`app/services/vllm_openai_translation_service.py:156`
```python
DEFAULT_CAST_ANCHOR = (
    "Yurie (the mother, she/her); the son (he/him); the tormentor (he/him)"
)
```
- Injected by `build_cast_anchor_line(cast=None)` (`:171`) → returns `"Cast: {body}"`, newlines flattened to guarantee ONE line.
- Consumed in `build_v11_context_prompt` (`:282`), inserted **between the instruction and the `Page:` block**, gated by the flag (`:299-301`):
  ```python
  cast_block = ""
  if getattr(settings, "translation_cast_anchor", False):
      cast_block = f"{build_cast_anchor_line()}\n\n"
  ```
- Flag defaults **False** (`app/config.py:355`). When off, the prompt is **byte-identical** to the trained v11 template — enforced by the golden test `tests/unit/test_cast_anchor_prompt.py:57` (`test_flag_off_is_byte_identical_to_golden`).
- Design contract (comments `:133-158`, config `:348-355`): the cast MUST be an **in-body context line, NEVER a `system` message** — a system message on this format-sensitive page-context path is the documented ~95% chrF++-collapse risk class (MEMORY: `feedback_chat_template_mismatch`). This constraint is load-bearing and non-negotiable.
- **Why disabled:** it was shipped as an unvalidated A/B lever (comment `:147-151` says "to be A/B'd on eval_pagecontext_heldout.jsonl"), never run/enabled, and the cast content is **guessed for one title** (Ikenie no Haha — Yurie is the only documented name; son/tormentor roles are "inferred conservatively from the title", `:153-155`). It is a scaffold, not a populated feature.

### 1b. `NAME_LOCKS` — output-side, POST-processing (different mechanism)

`app/services/name_glossary.py:161`
```python
NAME_LOCKS = (
    NameLock(canonical="Yurie", jp_kana="ユリエ",
             mis_romaji=("Julie","Lucia","Yulie","Yurié","Yuri")),
    NameLock(canonical="Ayumu", jp_kana="あゆむ",
             mis_romaji=("Ayuuuummm","Ayumumu","Aymu","Ayu")),
)
```
- Fires in `canonicalize_names()` **pass 3** (`:404-408`) only when the locked `jp_kana` appears in the OCR'd source; forces enumerated mis-romanizations of that name to one canonical spelling. Pure output repair — it does **not** touch the prompt.
- This module *explicitly forbids* prompt glossaries: `name_glossary.py:8` "**DO NOT add a glossary to the model prompt.** The v11 translation model is acutely train/serve format-sensitive (a documented ~95% quality collapse…)." This is the same landmine as 1a's design constraint, and it is why name-anchoring was pushed to the *output* while pronoun-anchoring (which cannot be repaired from output alone — you can't recover the referent's gender from an already-wrong "he") sits stuck in the disabled prompt lever.

**Format of both today:** free-text (`"Name (role, pronoun); …"`) for the anchor; a typed dataclass tuple (`canonical / jp_kana / mis_romaji`) for the locks. They share the *same* underlying entity data (name, canonical spelling) but store it twice and never talk to each other. A register unifies them.

### 1c. The injection seam already accepts a dynamic cast (important)

`build_cast_anchor_line(cast: str | None = None)` already takes a per-call cast string (`:171`). The **only** reason it's static is that `build_v11_context_prompt` calls it with no argument and there is no `cast` parameter threaded down from `translate_page_context_marked → _translate_one_marked → build_v11_context_prompt`. So injecting a *dynamic per-chapter* register is a parameter-threading change, not a new prompt surface — the byte-exact golden test still guards the flag-off path unchanged.

---

## 2. Does the batch pipeline carry chapter-level state? (verified)

**Yes — the state *home* exists; the *accumulation* does not.**

`scripts/batch_translate_chapter.py`:
- `class ChapterPipeline` (`:316`) is instantiated **once** (`:1022`, `pipe = ChapterPipeline()`).
- `main()` loops pages and calls `pipe.render_page(src, …)` **per page** (`:1027-1050`). Because `pipe` is one long-lived object, any `self.<attr>` set in one page persists to the next. Today `__init__` (`:319`) only holds stateless resources: `self.detector`, `self.ocr`, `self.translator` — **no cross-page character/cast state**.
- Every page already OCRs all dialogue: `page_context_lines` / `kept_texts` are built per page (`:553-565`) and fed to `translate_page_context_marked`. **The raw JP text needed to mine names/honorifics is already in hand on every page, for free.**

So a rolling cast register is a natural add: `self.cast_register` accumulated in the page loop, updated after each page's OCR, and threaded into the translate call. The batch pipeline is exactly where "full chapter context" is available and where the audit's "chapter-level rolling cast register (batch pipeline already has the state)" recommendation lands.

**The router/browser path has no such home.** `app/routers/translate.py` processes **one image per request**; its `session_id` (`:325,:359`) is an ephemeral per-connection uuid (`uuid.uuid4().hex[:8]` when absent), **not** a persistent title/chapter identity. There is no title id, no prior-page memory. This is the degradation boundary (§3d).

---

## 3. Design: the cast register

### 3a. Representation (single source of truth, replaces both tables)

An ordered list of character records, keyed by JP name/kana, carrying everything both current mechanisms need:

```python
@dataclass(frozen=True)
class CastMember:
    jp_kana: str                 # "ユリエ" — match key against OCR source
    canonical_en: str            # "Yurie" — feeds NAME_LOCKS output repair
    gender: str | None           # "female" | "male" | None(unknown)
    pronoun: str | None          # "she/her" | "he/him" | "they/them"
    role: str | None             # "the mother" (short, human/verified)
    aliases: tuple[str, ...]     # observed mis-romanizations -> NAME_LOCKS
    confidence: float            # register-build confidence, gates inclusion
    honorific_hints: tuple[str, ...]  # ("お母さん","ママ") evidence seen

CastRegister = list[CastMember]  # persisted per-title as JSON/YAML manifest
```

This subsumes `DEFAULT_CAST_ANCHOR` (render `canonical_en (role, pronoun)` clauses) **and** `NAME_LOCKS` (`jp_kana → canonical_en` + `aliases`). One table, two consumers — closes the "two disconnected tables" gap.

**Persistence:** a per-title manifest file, e.g. `data/cast_manifests/<title_slug>.json`. Batch runs load/refine/write-back; a curated manifest can be shipped for known titles.

### 3b. Building it — three tiers, cheapest first (all NON-training)

The register only needs `{name → gender/pronoun}` for the *recurring* cast — a handful of entries. Precision >> recall (a wrong gender is worse than an absent one).

- **Tier 0 — honorific/name mining from OCR'd JP (no extra model, runs inside the batch loop).**
  After each page's OCR, scan `page_context_lines` for gender-marking Japanese cues and attach them to co-occurring name tokens:
  - Kana/kanji name + honorific: 〜さん/ちゃん/くん/様; 兄/姉, お兄ちゃん/お姉ちゃん, お父さん/お母さん, 息子/娘, 彼/彼女, 王子/姫, etc. `くん` strongly ⇒ male, `姉/娘/母` ⇒ female, and so on. (The pipeline **already normalizes** these forms — see the onee-chan OCR-variant handling in `name_glossary.py:205`.)
  - Accumulate evidence counts across pages; a member is promoted to the active register once its gender evidence crosses a threshold and is self-consistent. This is the "rolling" refinement — early pages seed it, later pages confirm/correct.
  - Cheap, deterministic, no VLM. Weakness: pro-drop pages give no cue; ambiguous honorifics (さん) give none. That's fine — unknowns stay `None` and simply aren't asserted.

- **Tier 1 — one-time VLM pass over the first N pages (batch only, opt-in).**
  Feed the first N page images (the pipeline already builds page-image data URLs — `translate.py:417`, `ndarray_to_data_url`) to the vision model with a *structured extraction* prompt ("list recurring characters: JP name if written, apparent gender, role"). This is a **separate call**, NOT the translation prompt — so it is *outside* the format-sensitive train/serve contract and carries none of the collapse risk. Visual gender cues resolve exactly the pro-drop cases Tier 0 can't. Cost: N extra VLM calls per chapter, amortized once.

- **Tier 2 — curated manifest.**
  For known/licensed titles, ship a human-verified manifest. Highest precision, zero runtime cost. The current `DEFAULT_CAST_ANCHOR`/`NAME_LOCKS` become the first two hand-authored manifests.

Tiers compose: Tier 2 manifest (if present) seeds; Tier 0 mines continuously; Tier 1 backfills genders Tier 0 left `None`. Confidence gates what actually reaches the prompt.

### 3c. Injecting it — reuse the proven, byte-safe seam (no new prompt surface)

Thread a `cast: str | None` down the existing call chain so the register renders into the **already-tested** in-body `Cast:` line:

```
render_page (build cast string from self.cast_register)
  -> translate_page_context_marked(..., cast=cast_str)
    -> _translate_one_marked(..., cast=cast_str)
      -> build_v11_context_prompt(lines, k_idx, cast=cast_str)
        -> build_cast_anchor_line(cast_str)   # ALREADY accepts this arg (:171)
```

Rules that keep the train/serve contract intact:
1. **Same insertion point, same shape** as the existing anchor: one `"Cast: …"` line, newline-flattened, between instruction and `Page:`. The golden byte-identity test (`test_cast_anchor_prompt.py:57`) must still pass with `cast=None`/flag-off — i.e. an empty/absent register produces the *exact* trained template.
2. **Only assert what's known.** Render clauses only for members with a resolved pronoun above threshold; drop `role`/pronoun when `None`. Never emit "he/him?" or a guess.
3. **Cap the line length.** A 20-character chapter cannot dump 20 clauses — the line would dominate the prompt and drift out of distribution. Include only the top-K most-frequent speakers on/near the current page (K≈3–5), so the injected line stays the size the (proven-safe) static anchor already is.
4. **Never a system message. Never multi-line.** Enforced by `build_cast_anchor_line`'s existing newline-flatten and by keeping this the sole seam.

Because the flag-off / empty-register path is provably the trained template, **the ~95% collapse landmine is avoided by construction** — the same guarantee the shipped scaffold already has, extended to dynamic content of the same shape.

The output-side `NAME_LOCKS` consumer is fed from the same register (`jp_kana`/`aliases`), so name-spelling repair and pronoun conditioning stay in sync instead of drifting as two hand-maintained tables.

### 3d. Graceful degradation: batch (full chapter) vs browser (single page)

| Path | Register available? | Behaviour |
|---|---|---|
| **Batch** (`ChapterPipeline`) | Yes — Tier 0 rolling + optional Tier 1/2 | Full conditioning; register refines across pages; write-back to manifest |
| **Browser, known title** | If client sends a title id → load shipped/cached manifest (Tier 2) | Conditioning from manifest; no cross-page state needed |
| **Browser, unknown title, single page** | No chapter context | **Falls back to today's behaviour**: empty register ⇒ byte-identical trained template (zero regression), OR the static `DEFAULT_CAST_ANCHOR` if the flag is on for a known title. Optionally Tier-0-mine *within the single page* for any names visible on that page only (weak but non-negative). |

The degradation is safe precisely because the null register == the current golden prompt. The browser path loses the *chapter*-scope benefit (which is the whole point — a single page is exactly the underdetermined unit), but never regresses. To give the browser the batch benefit, the client would need to supply a stable title/chapter id so the server can key a per-title manifest cache — a clean, optional enhancement, not a prerequisite.

---

## 4. Why this can work where training failed

- **The failure is informational, not a model weakness.** JP is pro-drop: a bubble like 「行くよ」 carries no gender. Training the 4B model to *emit* "she" is asking it to hallucinate a fact absent from its input; the ceiling documented in `project_furube_human_eval` and `project_mt_finetuning_roadmap` ("4B is the discourse bottleneck", "pronoun ceiling") is the model correctly refusing to invent, or inventing wrong. The referent's gender is a **chapter-level fact**. A register *supplies the missing input* rather than training the model to guess it.
- **This is textbook MT entity/glossary conditioning.** Terminology/named-entity constraints injected into the prompt (or via constrained decoding) are the standard, well-established way to force consistent rendering of entities NMT/LLM-MT would otherwise vary — the same idea used for glossary terms, applied to the pronoun/gender attribute of a named referent. It moves a per-sentence ambiguity to a document-level constraint.
- **It's orthogonal and complementary to the #1 lever (image-as-context).** The image path (`translation_serve_image_context`, already wired — `translate.py:417`, `build_image_text_content` `:324`) helps the model *see* the speaker; the register helps it *name* the speaker's gender even when the panel doesn't show them (off-panel referent, back-of-head, narration about an absent character). Register also feeds the Tier-1 VLM extraction, so the two share machinery. Stacking both is plausibly additive on the pronoun_gender bucket.
- **Post-hoc name repair already proves the entity data helps.** `NAME_LOCKS` demonstrably stabilizes name spelling from the same `jp_kana → canonical` mapping; extending that verified data with a gender/pronoun field and feeding it to the *prompt* (the only place pronouns can be fixed) is the minimal next step, not a speculative new system.

---

## 5. Feasibility + biggest risks

**Feasibility: medium build, low serve risk.** The injection half is nearly done (seam + golden test exist; needs a `cast` param threaded through 3 functions and a top-K renderer). The persistent batch object exists. The real work is the register *builder* (Tier 0 miner + optional Tier 1 VLM pass + manifest I/O) and its validation harness.

**Risks, ranked:**
1. **Register-build accuracy (highest).** A wrong gender in the register poisons *every* bubble for that character — worse than today's per-bubble coin-flip. Mitigations: precision-over-recall thresholds; assert a pronoun only with strong, self-consistent evidence; keep the Tier-2 curated manifest as the trusted default for shipped titles; Tier-0 mining is *advisory* (raises confidence) not *authoritative* until it clears a bar. Validate on `eval_pagecontext_heldout.jsonl` (the A/B target the scaffold already names, config `:349`) on the pronoun_gender / mistranslation buckets — register-on must beat register-off *and* beat the static-anchor arm, or it doesn't ship.
2. **Prompt-format drift (medium, mostly mitigated).** Any dynamic content in the page-context prompt risks the collapse class. Mitigated by: same insertion point/shape as the proven static anchor, the flag-off/empty-register golden test, newline-flatten, top-K length cap, and never using a system message. The invariant to enforce in tests: *empty register ⇒ byte-identical trained template*; *non-empty ⇒ off-prompt plus exactly one `Cast:` line*. (Extend `test_cast_anchor_prompt.py`.)
3. **Latency on the browser path (medium).** Tier-1 VLM extraction is chapter-amortized and batch-only — never on the browser request path. Tier-0 mining is regex over already-OCR'd text (negligible). The only browser cost is loading a cached manifest (a dict lookup). The injected `Cast:` line adds a few dozen tokens to a prompt whose expensive shared prefix is already prefix-cached by vLLM (`:731`) — marginal. **No new latency on the hot single-page path.**
4. **Register staleness / role drift across a long chapter (low).** A character's role label can go stale (e.g. a reveal). Keep `role` optional and coarse; lean on `pronoun` (stable) over `role` (narrative). The rolling refinement corrects gender if early evidence was wrong, but a shipped conditioning that was wrong for pages already rendered can't be retroactively fixed in a streaming render — acceptable for batch (re-runnable), and the browser uses the trusted manifest.

**Recommended sequencing:** (a) thread the `cast` param + top-K renderer + extend the golden test; (b) hand-author the Ikenie manifest from the existing anchor/locks and A/B the *static-per-title* arm on the heldout set first — this de-risks injection with zero builder work; (c) only if (b) wins, build the Tier-0 miner in `ChapterPipeline` and measure the *rolling* arm; (d) Tier-1 VLM extraction last, gated on whether Tier-0 leaves too many genders unknown.

---

## Key files (absolute paths)

- `/home/danny/Documents/personal/extension/backend/app/services/vllm_openai_translation_service.py` — `DEFAULT_CAST_ANCHOR` :156, `build_cast_anchor_line` :171 (accepts dynamic cast), `build_v11_context_prompt` :282, injection gate :299-301, `_translate_one_marked` :629, `translate_page_context_marked` :706, image-context helper :324
- `/home/danny/Documents/personal/extension/backend/app/services/name_glossary.py` — `NAME_LOCKS` :161, `canonicalize_names` :357, "DO NOT add a glossary to the prompt" :8, per-title cast TODO :52-59/:155-160
- `/home/danny/Documents/personal/extension/backend/app/config.py` — `translation_cast_anchor` flag :355 (default False), A/B target note :349
- `/home/danny/Documents/personal/extension/backend/scripts/batch_translate_chapter.py` — `ChapterPipeline` :316, single instantiation :1022, per-page loop :1027, per-page OCR context :553-565
- `/home/danny/Documents/personal/extension/backend/app/routers/translate.py` — single-image path, ephemeral `session_id` :325/:359, page-image build :417
- `/home/danny/Documents/personal/extension/backend/tests/unit/test_cast_anchor_prompt.py` — byte-identity golden :57, injection-shape assertions :82-122
