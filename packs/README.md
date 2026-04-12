# Packs

Each pack is a directory named after its `id`. The directory contains:

```
packs/
└── <pack_id>/
    ├── pack.yaml
    ├── entity_names.tsv
    └── string_exonyms.tsv
```

---

## `pack.yaml`

Identity and behaviour manifest. Three sections:

**Identity** — `id`, `display_name`, `bcp47`, `version`, `kind`, `output_scripts`, `capabilities`. The `kind` field is one of `historical | modern | constructed | hybrid`.

**Eras** — named date ranges used by the game UI only. They do not affect resolution logic; validity dates on individual data rows do that. `default_era` is used when the caller provides no date.

**Pipeline** — ordered list of resolution steps. The engine tries steps in sequence and stops at the first step that returns candidates, unless a step carries `merge: true`, in which case later steps can still contribute lower-ranked candidates for top-k output. Step types: `entity_lookup`, `string_exonym`, `geonames_source`, `icu_transliteration`, `uroman`.

**Ranking** — additive integer bonuses and penalties applied to all candidates after the pipeline runs. Priority integers in the data files are tiebreakers *within* a single step's result set only; they are never compared across steps or against these ranking values.

---

## `entity_names.tsv`

Maps a named entity `(namespace, entity_id)` to an output name, optionally scoped to a date range. Fires via the `entity_lookup` pipeline step. Produces the highest-scoring candidates.

| Column | Required | Notes |
|--------|----------|-------|
| `namespace` | yes | `geonames` \| `wikidata` \| `osm_relation` \| `local` |
| `entity_id` | yes | Opaque ID within the namespace. For `geonames`, must be an integer present in the `geoname` table. |
| `output_name` | yes | The name to emit. |
| `valid_from` | no | `(-)YYYY-MM-DD`. Astronomical year convention (year 0 = 1 BCE). Empty = open. |
| `valid_to` | no | Same format. Empty = open. |
| `priority` | yes | Tiebreaker within overlapping rows for the same entity and date. Default 100. |
| `tags` | no | Comma-separated tokens. Stored one-per-row in `pack_entity_name_tag`. |
| `note` | no | Free text. Not used by the engine. |

Lines beginning with `#` are comments and are stripped by the importer.

Multiple rows for the same entity with overlapping date ranges are intentional: the highest-priority row is top-1; others appear in top-k.

---

## `string_exonyms.tsv`

Maps a normalised input string to an output name. Fires via the `string_exonym` pipeline step, after `entity_lookup` has failed or been skipped. Produces lower-scoring candidates than `entity_names.tsv`.

The importer normalises the `input` column (NFC, casefold, strip) on import; store the natural human-readable form in the file.

| Column | Required | Notes |
|--------|----------|-------|
| `input` | yes | Raw input string. Normalised by the importer. |
| `output_name` | yes | The name to emit. |
| `valid_from` | no | Same format as `entity_names.tsv`. |
| `valid_to` | no | Same format. |
| `priority` | yes | Tiebreaker within overlapping rows for the same input and date. Default 100. |
| `note` | no | Free text. Not used by the engine. |

Lines beginning with `#` are comments and are stripped by the importer.

**When to use this file vs. `entity_names.tsv`:** if you know the GeoNames or Wikidata ID, always prefer `entity_names.tsv`. It produces higher-scoring candidates and is immune to input-string variation. Use `string_exonyms.tsv` only when no canonical entity ID is available.