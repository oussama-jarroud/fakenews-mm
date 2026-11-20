"""End-to-end mock test for Kaggle 01 Prepare pipeline."""

import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from PIL import Image

def test_full_script():
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Create a dummy TSV file mimicking Fakeddit
        tsv_path = os.path.join(temp_dir, "multimodal_train.tsv")
        img_dir = os.path.join(temp_dir, "raw_server_images")
        os.makedirs(img_dir, exist_ok=True)

        # Create 10 dummy images
        urls = []
        for i in range(10):
            p = os.path.join(img_dir, f"test_{i}.jpg")
            # Create pairs that share images:
            # 0, 1 share same image pattern
            # 2, 3 share same image pattern
            # 4 is different
            # 5 is broken url
            arr = np.full((100, 100), fill_value=(i % 3) * 80, dtype=np.uint8)
            Image.fromarray(arr).save(p, "JPEG")
            # file:// URL for local testing
            urls.append(f"file:///{p.replace(os.sep, '/')}")
        urls[5] = "file:///invalid/path/that/does/not/exist.jpg"

        data = {
            "id": [f"post_{i}" for i in range(10)],
            "clean_title": [f"News Headline sample number {i}" for i in range(10)],
            "image_url": urls,
            "2_way_label": [i % 2 for i in range(10)],
            "6_way_label": [i % 6 for i in range(10)],
            "subreddit": ["fake_sub" if i % 2 == 0 else "real_sub" for i in range(10)],
            "hasImage": [True] * 10
        }
        pd.DataFrame(data).to_csv(tsv_path, sep="\t", index=False)
        print("Created synthetic TSV.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_full_script()

