"""Token attribution and importance module (Member B - Task B4 / Contribution C3).

Implements occlusion-based token attribution: measures prediction drop when each word
in the headline is masked out.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_token_attribution(
    text: str,
    predict_fn,
    image_url: Optional[str] = None,
) -> Dict[str, float]:
    """Computes attribution score for each word in the text using occlusion.

    Args:
        text: Original input headline text.
        predict_fn: Callable taking (text, image_url) and returning dict with 'confidence'.
        image_url: Accompanying image URL or None.

    Returns:
        Dict mapping word -> attribution impact score (higher means word is more critical).
    """
    words = text.split()
    if len(words) <= 1:
        return {text: 1.0}

    # Baseline prediction with full text
    base_res = predict_fn(text=text, image_url=image_url)
    base_conf = base_res.get("confidence", 0.5)

    attributions = {}
    for i, word in enumerate(words):
        # Occlude (remove) word i
        occluded_words = [w for j, w in enumerate(words) if j != i]
        occluded_text = " ".join(occluded_words)

        occluded_res = predict_fn(text=occluded_text, image_url=image_url)
        occluded_conf = occluded_res.get("confidence", 0.5)

        # Impact = change in confidence
        impact = abs(base_conf - occluded_conf)
        attributions[word] = round(float(impact), 4)

    # Normalize scores to [0.0, 1.0]
    max_val = max(attributions.values()) if attributions else 1.0
    if max_val > 0:
        attributions = {k: round(v / max_val, 4) for k, v in attributions.items()}

    return attributions
