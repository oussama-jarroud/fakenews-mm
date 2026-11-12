# Beyond Shortcuts: Multimodal Fake News Detector

## Day 1 Scope Decisions
* **D1 Falsification target:** Misleading image-text pairs (false connection, out-of-context, manipulated).
* **D2 Label granularity:** 2-way main, 6-way secondary.
* **D3 Datasets:** Fakeddit subset + own hard-mismatch pairs + one external test.
* **D4 Language:** English.
* **D5 Outside evidence:** None in the core.
* **D6 Backbone:** Frozen CLIP ViT-B/32 + BERT/ResNet50 baselines.
* **D7 Intent (mis/dis/mal):** Out of scope.