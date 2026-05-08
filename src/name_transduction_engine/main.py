from name_transduction_engine.datasets.dataset_provider import ensure_datasets
from name_transduction_engine.enrichment.enrichment_provider import (
    ensure_builtin_pack_enrichment,
)
from name_transduction_engine.paths import DB_PATH_NAMES, BUILTIN_PACKS_DIR


def main():
    ensure_datasets()
    ensure_builtin_pack_enrichment(DB_PATH_NAMES, BUILTIN_PACKS_DIR)


if __name__ == "__main__":
    main()
