# Third-party notices

This file records third-party code, models, tokenizers, data, and other assets
bundled with or referenced by this repository. Those materials remain subject
to their own terms.

| Path | Upstream project or asset | License information |
| --- | --- | --- |
| `fairseq/` | [Fairseq](https://github.com/facebookresearch/fairseq) | `fairseq/LICENSE` (MIT) |
| `SimulEval/` | [SimulEval](https://github.com/facebookresearch/SimulEval) | `SimulEval/LICENSE` (CC BY-SA 4.0) |
| `SHAS/` | [SHAS](https://github.com/mt-upc/SHAS) | `SHAS/LICENSE` (MIT) |
| `mwerSegmenter/` | RWTH mwerSegmenter | `mwerSegmenter/LICENSE` (noncommercial terms) |
| `iwslt/speech_segmentation/en_shas_model.pt` | SHAS English segmentation checkpoint | See the SHAS project documentation; SHA-256: `11a7bd157fd60b0a236438586f857f63df705d06854c5db08f0acc39ae834ae4` |
| `iwslt/tokenizer/` | SentencePiece and LLM-derived tokenizer assets | Subject to the applicable upstream model and dataset terms |

Dependencies installed through `pyproject.toml` remain subject to their own
licenses. Before redistributing a derived release, verify the terms that apply
to every third-party component and asset included in that release.
