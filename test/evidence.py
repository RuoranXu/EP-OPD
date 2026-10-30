from __future__ import annotations
from typing import Any
import re
SPACE_RE=re.compile(r"\s+")

def normalize(value: Any) -> str:
    return SPACE_RE.sub(' ', str(value or '')).strip().casefold()

def block_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {'META-TITLE': {'block_id': 'META-TITLE', 'kind': 'title', 'text': str(row.get('title') or '')}, 'META-ABSTRACT': {'block_id': 'META-ABSTRACT', 'kind': 'abstract', 'text': str(row.get('abstract') or '')}}
    for (section_index, section) in enumerate(row.get('sections') or []):
        section_id = str(section.get('section_id') or f'S{section_index + 1}')
        heading_id = f'{section_id}-H'
        blocks[heading_id] = {'block_id': heading_id, 'kind': 'heading', 'section_id': section_id, 'section_name': str(section.get('section_name') or ''), 'text': str(section.get('section_name') or '')}
        for paragraph in section.get('paragraphs') or []:
            paragraph_id = str(paragraph.get('paragraph_id') or '')
            if paragraph_id:
                blocks[paragraph_id] = {'block_id': paragraph_id, 'kind': 'paragraph', 'section_id': section_id, 'section_name': str(section.get('section_name') or ''), 'text': str(paragraph.get('text') or '')}
    for item in row.get('figures_and_tables') or []:
        float_id = str(item.get('float_id') or '')
        if float_id:
            blocks[float_id] = {'block_id': float_id, 'kind': 'float', 'text': str(item.get('caption') or '')}
    return blocks

def ordered_blocks(row: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = block_map(row)
    order = ['META-TITLE', 'META-ABSTRACT']
    for section in row.get('sections') or []:
        section_id = str(section.get('section_id') or '')
        order.append(f'{section_id}-H')
        order.extend((str(paragraph.get('paragraph_id')) for paragraph in section.get('paragraphs') or []))
    order.extend((str(item.get('float_id')) for item in row.get('figures_and_tables') or []))
    return [mapping[block_id] for block_id in order if block_id in mapping and normalize(mapping[block_id]['text'])]
