"""Tests for text mask generation – only translate/remove regions included."""

from __future__ import annotations

import numpy as np
import pytest

from image_translation.models.text_region import TextAction, TextRegion
from image_translation.imaging.text_mask import create_text_mask


class TestTextMask:
    def test_only_translate_and_remove(self):
        """Mask should only cover translate and remove regions."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)

        regions = [
            TextRegion(
                id="t1", source_text="translate me",
                confidence=0.9,
                polygon=[[10, 10], [100, 10], [100, 50], [10, 50]],
                action=TextAction.translate,
            ),
            TextRegion(
                id="t2", source_text="preserve me",
                confidence=0.9,
                polygon=[[10, 60], [100, 60], [100, 100], [10, 100]],
                action=TextAction.preserve,
            ),
            TextRegion(
                id="t3", source_text="remove me",
                confidence=0.9,
                polygon=[[150, 10], [250, 10], [250, 50], [150, 50]],
                action=TextAction.remove,
            ),
            TextRegion(
                id="t4", source_text="review me",
                confidence=0.5,
                polygon=[[150, 60], [250, 60], [250, 100], [150, 100]],
                action=TextAction.review,
            ),
        ]

        mask = create_text_mask(img, regions, expansion_pixels=0)

        # Translate region should be white
        assert mask[30, 50] == 255
        # Preserve region should be black
        assert mask[80, 50] == 0
        # Remove region should be white
        assert mask[30, 200] == 255
        # Review region should be black
        assert mask[80, 200] == 0

    def test_expansion_pixels(self):
        """Mask should expand with positive expansion_pixels."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        regions = [
            TextRegion(
                id="t1", source_text="x",
                confidence=0.9,
                polygon=[[50, 50], [60, 50], [60, 60], [50, 60]],
                action=TextAction.translate,
            ),
        ]

        mask_small = create_text_mask(img, regions, expansion_pixels=0)
        mask_large = create_text_mask(img, regions, expansion_pixels=5)

        assert np.sum(mask_large > 0) > np.sum(mask_small > 0)
