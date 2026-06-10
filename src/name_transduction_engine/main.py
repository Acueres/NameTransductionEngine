from name_transduction_engine.datasets.dataset_provider import ensure_datasets
from name_transduction_engine.enrichment.enrichment_provider import (
    ensure_builtin_pack_enrichment,
)


def main():
    ensure_datasets()
    ensure_builtin_pack_enrichment()


if __name__ == "__main__":
    main()
