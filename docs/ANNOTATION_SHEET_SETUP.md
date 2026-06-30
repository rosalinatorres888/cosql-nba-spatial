# WOZ Annotation Sheet Setup Guide

**CoSQL NBA Spatial · IE7600 · Real-Time Collaboration**

---

## ✅ Setup Complete — Live Resources

| Resource | Link | Status |
|----------|------|--------|
| **Annotation Guide (Google Doc)** | [T7a Annotation Protocol](https://docs.google.com/document/d/1BnUD5ly3QYznIl77ce8zN8G7JgkvgqRRHi4HpJvE580/edit?tab=t.0) | ✅ Live |
| **Annotation Tracker (Google Sheets)** | [CoSQL NBA WOZ Annotations](https://docs.google.com/spreadsheets/d/1mCtGq4VJPBN4ow6oAIkR58xeSMXgS_RmtTvAZgLJJJ0/edit?gid=0#gid=0) | ✅ Live |
| **Review Tool** | [nba-cosql-spatial-annotation-tool.netlify.app](https://nba-cosql-spatial-annotation-tool.netlify.app/) | ✅ Live |
| **Annotation Guide (repo)** | `/docs/query_classes_and_clarify_templates.md` | ✅ Local |

---

## 📊 Current Status (as of Jun 30, 2026)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Total WOZ pairs | ≥ 120 | **139** | ✅ Done |
| Approved | ≥ 95% | **138 / 139 (99%)** | ✅ Done |
| Flagged / needs review | 0 | **1** | ⚠️ Resolve before M2 |
| Execution pass | ≥ 95% | **All verified live** | ✅ Done |
| Cohen's κ | ≥ 0.75 | Calculate before Jul 3 | ⏳ Pending |

### Distribution by Query Class

| Class | CSV File | Pairs | Target |
|-------|----------|-------|--------|
| Spatial Zone | `annotation_batch_class1_spatial_zone.csv` | 18 | ≥ 15 ✅ |
| Temporal Scope | `annotation_batch_class2_temporal_scope.csv` | 18 | ≥ 15 ✅ |
| Player/Entity | `annotation_batch_class3_player_entity.csv` | 18 | ≥ 15 ✅ |
| Simple Aggregation | `annotation_batch_class4_simple_aggregation.csv` | 18 | ≥ 15 ✅ |
| Comparative Aggregation | `annotation_batch_class5_comparative_aggregation.csv` | 18 | ≥ 15 ✅ |
| Multi-Turn Coreference | `annotation_batch_class6_coreference.csv` | 16 | ≥ 15 ✅ |
| Game/Matchup Context | `annotation_batch_class7_game_context.csv` | 18 | ≥ 15 ✅ |
| Shot Characteristics | `annotation_batch_class8_shot_characteristics.csv` | 15 | ≥ 15 ✅ |

> **Note:** Class 6 (Multi-Turn Coreference) has a different schema — includes `turn_index` and `prior_utterance` columns. Account for this when merging to final CSV.

---

## 🚨 M2 Checklist (Due Jul 3)

- [ ] **Resolve the 1 flagged pair** — open the review tool, find it, fix or exclude it
- [ ] **Calculate Cohen's κ** on state labels across all 139 pairs (target ≥ 0.75)
- [ ] **Update `kappa_agreement` column** in each batch CSV with final κ value
- [ ] **Merge all 8 batch CSVs** into one final file (see Export section below)
- [ ] **Commit final CSV** to repo and tag as `m2-submission-woz-final`
- [ ] **Verify execution pass rate** ≥ 95% on merged file

---

## Exporting to Final CSV (M2 Submission, Jul 3)

### Step 1: Merge the 8 batch CSVs

```bash
# Run from repo root
cd annotation

# Classes 1-5, 7-8 share the standard schema
head -1 annotation_batch_class1_spatial_zone.csv > ../woz_annotation_pairs_final.csv

for f in annotation_batch_class1_spatial_zone.csv \
          annotation_batch_class2_temporal_scope.csv \
          annotation_batch_class3_player_entity.csv \
          annotation_batch_class4_simple_aggregation.csv \
          annotation_batch_class5_comparative_aggregation.csv \
          annotation_batch_class7_game_context.csv \
          annotation_batch_class8_shot_characteristics.csv; do
  tail -n +2 "$f" >> ../woz_annotation_pairs_final.csv
done

# Class 6 (coreference) has extra columns — append separately and note in submission
tail -n +2 annotation_batch_class6_coreference.csv >> ../woz_annotation_pairs_final_coreference.csv
```

### Step 2: Verify counts

```bash
# Total pairs (should be ≥ 120)
wc -l woz_annotation_pairs_final.csv

# Execution pass rate
grep "TRUE" woz_annotation_pairs_final.csv | wc -l

# Per-class counts
cut -d',' -f3 woz_annotation_pairs_final.csv | sort | uniq -c
```

### Step 3: Commit and tag

```bash
git add annotation/woz_annotation_pairs_final.csv
git commit -m "[T7a] Final WOZ annotation set — 139 pairs, κ=___, 99% approved"
git push origin main

git tag m2-submission-woz-final
git push origin m2-submission-woz-final
```

---

## Team Roles & Access Control

### Role Definitions

| Person | Role | Model | Sheet Access | Responsibilities |
|--------|------|-------|--------------|-----------------|
| **Rosalina** | SQL Expert + Data Controller | CoSQL (yours) | ✅ Edit | Gold SQL, query_class, state labels, all final decisions |
| **Sean** | NL User | Transformer attention | 👁️ View only | Submit NL questions via Form only |
| **Craig** | NL User | GRU seq2seq | 👁️ View only | Submit NL questions via Form only |

### Why This Structure

Craig and Sean are **NL users only** — their job is to ask questions the way a real user would, with no knowledge of the gold SQL or query class taxonomy. This is correct WOZ protocol: the wizard (Rosalina) controls the SQL side entirely; users just talk naturally.

**Craig and Sean must not:**
- See the gold SQL before M2 submission (contaminates model evaluation)
- Have edit access to the annotation tracker (risk of accidental or uninformed changes)
- Know which query class their question belongs to (biases phrasing)
- Be given example questions to copy (produces fabricated, non-natural data)

### NL Submission Process (Craig & Sean)

1. Craig/Sean think of a natural question they'd ask about Celtics NBA data
2. They submit it via Google Form (link in README) — NL question + any context only
3. Rosalina reviews the submission, writes gold SQL, verifies execution, assigns class and state
4. Pair enters the tracker under Rosalina's name as annotator

### Model Evaluation Integrity

Since Craig and Sean are both **annotators and model authors**, evaluation must be structured to avoid bias:

- **Test set is frozen by Rosalina** — neither Craig nor Sean selects which pairs evaluate their model
- **Cross-evaluation:** Craig's model is scored on pairs where Sean wrote the NL; Sean's model is scored on pairs where Craig wrote the NL
- **Rosalina's model** is evaluated on both (no conflict — she wrote no NL utterances)
- Final comparison uses the same held-out subset for all three models

---

## Annotation Workflow (Reference — Active Jun 15–Jun 30)

### Pair Workflow

**Step 1 — NL User (Craig or Sean):** Submits a natural language question via Form. No SQL, no class taxonomy visible.

**Step 2 — SQL Expert (Rosalina):** Reviews submission, writes `gold_sql`, verifies execution on live PostgreSQL DB, assigns `query_class` and `state`, enters in tracker.

**Step 3 — Agreement check:** Every 20 pairs, calculate Cohen's κ on state labels. If κ < 0.75, re-annotate pairs below threshold.

---

## Annotation Tips & Gotchas

### Writing Good Gold SQL

✓ **DO:**
- Test every SQL on the live database before submitting
- Use `player_id`, not player name (names are not unique keys)
- Quote string literals (`'BOS'`, `'LAL'`)
- Use `ILIKE '%name%'` for player lookups to handle name variations
- Comment complex queries: `-- Filter for Q4 and paint zone`

✗ **DON'T:**
- Hardcode `player_id` without confirming it matches `nba_api`
- Use undefined columns (check `schema.sql` first)
- Write SQL that could timeout (avoid full table scans if possible)
- Assume zone names — always verify against lexicon in `query_classes_and_clarify_templates.md`

### Common Errors

| Error | Symptom | Fix |
|-------|---------|-----|
| Wrong zone bounds | "Left corner" mapped to `x ≥ 220` (right corner) | Left corner: `x <= -220 AND y <= 90` |
| `game_clock` vs. `shot_clock` | "With 5 seconds left" ambiguous | Clarify which timer before writing SQL |
| Missing JOINs | "Celtics-Lakers game" returns no results | Join `games` table on `game_id` |
| Entity alias | "JT" not recognized | Resolve to full name via `players` table |
| Integer division | FG% returns 0 | Use `CAST(... AS FLOAT)` |

---

**Document Version:** v2.0  
**Last Updated:** Jun 30, 2026  
**Status:** ✅ Annotation complete — M2 export pending
