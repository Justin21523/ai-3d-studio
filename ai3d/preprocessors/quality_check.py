"""Image quality checker — resolution, aspect ratio, blur, and coverage checks."""
from __future__ import annotations

from pathlib import Path

from ai3d.core.models import QualityCheckResult

_MIN_RESOLUTION = 256
_MAX_ASPECT_RATIO = 4.0
_BLUR_VARIANCE_THRESHOLD = 100.0


class QualityChecker:
    def check(self, image_path: Path) -> QualityCheckResult:
        try:
            from PIL import Image, ImageFilter  # type: ignore[import]
            import numpy as np  # type: ignore[import]
        except ImportError as exc:
            return QualityCheckResult(
                passed=False,
                score=0.0,
                issues=[f"Missing dependency: {exc}"],
            )

        issues: list[str] = []

        try:
            img = Image.open(image_path)
        except Exception as exc:
            return QualityCheckResult(
                passed=False,
                score=0.0,
                issues=[f"Cannot open image: {exc}"],
            )

        w, h = img.size
        has_alpha = img.mode == "RGBA"

        # Resolution check
        if w < _MIN_RESOLUTION or h < _MIN_RESOLUTION:
            issues.append(f"Image too small: {w}x{h} (minimum {_MIN_RESOLUTION}px per side).")

        # Aspect ratio check
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > _MAX_ASPECT_RATIO:
            issues.append(f"Extreme aspect ratio: {aspect:.1f}:1 (maximum {_MAX_ASPECT_RATIO}:1).")

        # Blur estimation via Laplacian variance
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        laplacian = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        blur_variance = float(laplacian.var())
        if blur_variance < _BLUR_VARIANCE_THRESHOLD:
            issues.append(f"Image may be blurry (edge variance: {blur_variance:.1f}).")

        # Subject coverage (non-transparent area ratio)
        coverage = 0.0
        if has_alpha:
            alpha = np.array(img.split()[3], dtype=np.float32)
            coverage = float((alpha > 10).sum()) / max(w * h, 1)
            if coverage < 0.05:
                issues.append(f"Very low subject coverage: {coverage:.1%} of image area.")
        else:
            # Estimate coverage by non-white pixels
            rgb = np.array(img.convert("RGB"), dtype=np.float32)
            non_white = ((rgb < 240).any(axis=2)).sum()
            coverage = float(non_white) / max(w * h, 1)

        # Score: 1.0 - (issue_count * penalty)
        score = max(0.0, 1.0 - len(issues) * 0.25)

        return QualityCheckResult(
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            image_width=w,
            image_height=h,
            has_alpha=has_alpha,
            estimated_subject_coverage=coverage,
        )
