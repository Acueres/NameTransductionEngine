from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR_GEONAMES = PROJECT_ROOT / "data" / "raw" / "geonames"
RAW_DIR_WIKIDATA = PROJECT_ROOT / "data" / "raw" / "wikidata"
DB_PATH_NAMES = PROJECT_ROOT / "data" / "names.sqlite"
BUILTIN_PACKS_DIR=PROJECT_ROOT / "packs"