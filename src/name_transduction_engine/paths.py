from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR_GEONAMES = PROJECT_ROOT / "data" / "raw" / "geonames"
RAW_DIR_WIKIDATA = PROJECT_ROOT / "data" / "raw" / "wikidata"
RAW_DIR_LANGUAGE_CODES = PROJECT_ROOT / "data" / "raw" / "language_codes"
BUILD_DIR = PROJECT_ROOT / "data" / "build"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

DB_PATH = PROJECT_ROOT / "data" / "names.sqlite"
BUILTIN_PACKS_DIR = PROJECT_ROOT / "packs"

# Optional local copies of dated dumps (`nte data fetch wikidata-raw`), named
# wikidata-YYYYMMDD-all.json.bz2. The build streams the dump from the network
# unless given a file with --source
WIKIDATA_RAW_DUMP_GLOB = "wikidata-*-all.json.bz2"
# Published compact dataset downloaded from GitHub (`nte data fetch wikidata`)
WIKIDATA_DOWNLOAD_DIR = RAW_DIR_WIKIDATA / "compact"
# Wikidata classes the build keeps (`nte data build wikidata-classes`)
WIKIDATA_CLASSES_PATH = RAW_DIR_WIKIDATA / "wikidata_classes.tsv"

# Streaming build (`nte data build wikidata-compact`)
WIKIDATA_BUILD_DIR = BUILD_DIR / "wikidata"
WIKIDATA_WORK_DIR = WIKIDATA_BUILD_DIR / "work"  # checkpoint and shards
WIKIDATA_COMPACT_DIR = WIKIDATA_BUILD_DIR / "compact"  # finished dataset

# Models learned from the datasets (built by `nte init`, after the datasets)
PERSIAN_MODEL_PATH = MODELS_DIR / "persian.json.gz"
ARABIC_MODEL_PATH = MODELS_DIR / "arabic.json.gz"
