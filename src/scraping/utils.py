from collections import Counter
from typing import List
import difflib
from difflib import SequenceMatcher


def tokenize_text(text:str) -> List[str]:
    """
    Args :
        - text : str -> The text to naively tokenize
    Returns :
        - list -> Naive tokenized text
    """
    return text.split('\n')


def find_redundant_segments(pages:list) -> list:
    """
    This functions utilize Counter() class to count the number of elements, if they are 'too present' then we assume they are
    headers or footers informations (not useful). To further remove them from the text content.

    Args :
        - pages:list -> The content of the previously parsed html pages
    Returns :
        - redundant_segments:lisdt -> The assumed headers and footers
    """

    segment_counter = Counter()

    for page in pages:
        print(page)
        segments = tokenize_text(page)
        segment_counter.update(segments)

    threshold = len(pages) * 0.5 # Redundancy Threshold
    redundant_segments = {segment for segment, count in segment_counter.items() if count > threshold} # Get all segments of the text that satisty the threshold condition

    return list(redundant_segments)

def remove_redundant_segments(pages:list, redundant_segments:list) -> list:
    """
    This function simply remove the previously gathered redundant segments from the original html extracted text
    """

    cleaned_html_content = []

    for text in pages:
        segments = tokenize_text(text)
        filtered_segments = [segment for segment in segments if segment not in redundant_segments]
        cleaned_html_content.append('\n'.join(filtered_segments))
    
    return cleaned_html_content

def handle_near_duplicates(pages: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """
    This function handles near duplicates by removing them from the list based on a similarity threshold.

    Args:
        - pages: List:str -> The list of text entries to process.
        - similarity_threshold: float -> The threshold above which two entries are considered near duplicates.

    Returns:
        - List:str -> The list with near duplicates removed.
    """
    unique_pages = []
    for page in pages:
        is_duplicate = False
        for unique_page in unique_pages:
            similarity = SequenceMatcher(None, page, unique_page).ratio()
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_pages.append(page)
    return unique_pages
