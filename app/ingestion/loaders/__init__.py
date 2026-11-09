from pathlib import Path
from typing import Callable, Dict

from app.ingestion.loaders.pdf import load_pdf
from app.ingestion.loaders.pptx import load_pptx
from app.ingestion.loaders.tabular import load_csv
from app.ingestion.loaders.text import load_text
from app.ingestion.models import ExtractedDocument

# source_type -> fonction de chargement. Phase 2 : image, ocr ; phase 3 : audio, web.
LOADERS: Dict[str, Callable[[Path], ExtractedDocument]] = {
    "pdf": load_pdf,
    "pptx": load_pptx,
    "csv": load_csv,
    "text": load_text,
}
