"""Conservative text cleaning preserving headings and paragraph boundaries."""

from __future__ import annotations

import re
import unicodedata

from .models import EnterpriseDocument


class DocumentCleaner:
    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean(self, document: EnterpriseDocument) -> EnterpriseDocument:
        cleaned = self.clean_text(document.text)
        pages = [type(page)(page.number, self.clean_text(page.text)) for page in document.pages]
        return EnterpriseDocument(
            document_id=document.document_id,
            text=cleaned,
            title=document.title,
            source=document.source,
            metadata=dict(document.metadata),
            pages=pages,
        )

