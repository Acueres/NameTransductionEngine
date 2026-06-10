from typing import Final
from name_transduction_engine.datasets.shared import get_and_save_file, build_session

from name_transduction_engine.paths import WIKIDATA_LOCATIONS_PATH

WIKIDATA_LOCATIONS_URL: Final[str] = (
    "http://github.com/Acueres/NameTransductionEngine/releases/download/data/wikidata_locations.jsonl.gz"
)


def download_wikidata_locations_data(force: bool = False) -> None:
    print("Wikidata locations download started.")
    WIKIDATA_LOCATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if WIKIDATA_LOCATIONS_PATH.exists() and not force:
        print("Skipping wikidata locations: already exists.")
        return

    with build_session() as session:
        get_and_save_file(
            session=session,
            url=WIKIDATA_LOCATIONS_URL,
            output_path=WIKIDATA_LOCATIONS_PATH,
        )

    print("Wikidata locations download finished.")
