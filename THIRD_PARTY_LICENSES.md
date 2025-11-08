# Third-Party Licenses

This project is distributed under the terms of the GNU Affero General Public License v3.0 (AGPLv3). The dependencies listed below introduce additional attribution or dual-licensing considerations. Please consult their upstream repositories for the authoritative terms before redistribution or commercial deployment.

| Component | License | Notes / Source |
|-----------|---------|----------------|
| **PyMuPDF** (`pymupdf` ≥ 1.26.6) | Dual License: AGPLv3 _or_ Artifex Commercial License | AGPL text bundled with the package; commercial licenses available from [Artifex](https://artifex.com/licensing/). |
| **pymupdf4llm** ≥ 0.1.8 | Dual License: AGPLv3 _or_ Artifex Commercial License | Ships with `pymupdf4llm-<version>.dist-info/licenses/LICENSE` (AGPLv3). Requires PyMuPDF. |
| **tabulate** | MIT License | SPDX: MIT. See [pypi.org/project/tabulate](https://pypi.org/project/tabulate/) for the full text. |
| **PDFMathTranslate (upstream fork)** | AGPLv3 | Original project at [Byaidu/PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate) distributes under AGPLv3; this fork inherits the same terms and must retain upstream copyright notices. |

When redistributing binaries (e.g., Docker images or packaged executables):

1. Include the AGPLv3 text (already present in `LICENSE`) and the notices above.
2. If you choose the Artifex commercial route for PyMuPDF / pymupdf4llm, ensure your downstream documentation reflects the commercial license terms and retain proof of purchase.
3. Provide source access as required by AGPLv3 when hosting network services based on this project.

For questions about commercial licensing of PyMuPDF or pymupdf4llm, contact Artifex Software at [support@artifex.com](mailto:support@artifex.com). For upstream project licensing matters, consult the original PDFMathTranslate repository and include its attribution when reusing branding or assets.
