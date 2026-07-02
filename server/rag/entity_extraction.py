"""
Entity Extraction — spaCy en_core_web_sm (free, local).
Auto-downloads the model if not installed.
"""
import subprocess
from typing import List

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[entity_extraction] Downloading en_core_web_sm …")
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> List[str]:
    """
    Extract unique named entities from *text* using spaCy.
    Text is truncated to 1 000 chars for speed.
    """
    nlp = _get_nlp()
    doc = nlp(text[:1000])
    entities = list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2})
    return entities