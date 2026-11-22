"""Production image preprocessing pipeline for OMR answer sheets.

Handles scan, photo, and hard conditions with:
- Feature-matching (SIFT + RANSAC) homography against canonical reference sheet
- Page contour quadrilateral detection fallback
- Deskewing
- Shadow normalization (background division + CLAHE)
- Denoising & adaptive thresholding
- Alignment confidence score calculation (flags LOW_ALIGNMENT_CONFIDENCE)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Structured result of the preprocessing pipeline."""
    canonical_bgr: np.ndarray
    canonical_gray: np.ndarray
    normalized_gray: np.ndarray
    thresholded: np.ndarray
    confidence: float
    flags: List[str] = field(default_factory=list)
    method_used: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        return "LOW_ALIGNMENT_CONFIDENCE" not in self.flags


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has min (x + y)
    rect[2] = pts[np.argmax(s)]  # Bottom-right has max (x + y)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has min (x - y)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has max (x - y)
    return rect


class SheetPreprocessor:
    """Robust OMR sheet preprocessor."""

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        canonical_size: Tuple[int, int] = (config.CANONICAL_WIDTH, config.CANONICAL_HEIGHT),
        confidence_threshold: float = config.ALIGNMENT_CONFIDENCE_THRESHOLD,
        reference: Optional[str] = None,
    ):
        # Reference sheet for feature alignment: "auto" uses the real p01-f01 scan
        # when the dataset is available, else a synthetic sheet rendered from the
        # template; "synthetic" always uses the template rendering (used by tests).
        self.reference = (reference or os.getenv("OMR_REFERENCE_SHEET", "auto")).lower()
        self.template_path = Path(template_path) if template_path else config.TEMPLATE_PATH
        self.canonical_width, self.canonical_height = canonical_size
        self.confidence_threshold = confidence_threshold

        # Load template
        self.template = self._load_template()
        self.anchors = self.template.get("alignment_anchors", {})

        # SIFT feature detector (optimized for speed and descriptor count)
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # Precompute reference data
        self._ref_bgr: Optional[np.ndarray] = None
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_kp: Optional[List[cv2.KeyPoint]] = None
        self._ref_des: Optional[np.ndarray] = None
        self._anchor_templates: Dict[str, np.ndarray] = {}
        self._init_reference()

    def _load_template(self) -> Dict[str, Any]:
        if self.template_path.exists():
            with open(self.template_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _init_reference(self):
        """Initialize reference sheet and anchor templates."""
        ref_candidate = config.VISION_IMAGES_DIR / "p01" / "p01-f01-scan.jpg"
        try:
            if self.reference != "synthetic" and ref_candidate.exists():
                raw = cv2.imread(str(ref_candidate))
                if raw is not None:
                    self._ref_bgr = cv2.resize(raw, (self.canonical_width, self.canonical_height))
                    self._ref_gray = cv2.cvtColor(self._ref_bgr, cv2.COLOR_BGR2GRAY)
                    self._ref_kp, self._ref_des = self.sift.detectAndCompute(self._ref_gray, None)
                    self._extract_anchor_templates(self._ref_gray)
                    return
        except (PermissionError, OSError):
            pass

        # Fallback synthetic sheet
        self._ref_gray = self._synthesize_reference_sheet()
        self._ref_bgr = cv2.cvtColor(self._ref_gray, cv2.COLOR_GRAY2BGR)
        self._ref_kp, self._ref_des = self.sift.detectAndCompute(self._ref_gray, None)
        self._extract_anchor_templates(self._ref_gray)

    def _extract_anchor_templates(self, gray: np.ndarray):
        """Extract anchor template patches from reference canonical sheet."""
        for name, roi in self.anchors.items():
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            patch = gray[y:y+h, x:x+w].copy()
            if patch.size > 0:
                self._anchor_templates[name] = patch

    def _synthesize_reference_sheet(self) -> np.ndarray:
        """Create clean synthetic canonical sheet with anchors, bubbles, and text."""
        canvas = np.full((self.canonical_height, self.canonical_width), 255, dtype=np.uint8)

        # Draw anchor markers (black square with high-contrast inner feature)
        for anchor in self.anchors.values():
            x, y, w, h = anchor["x"], anchor["y"], anchor["w"], anchor["h"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), 0, -1)
            # Inset white inner rectangle to ensure non-zero template variance
            iw = max(2, w // 4)
            ih = max(2, h // 4)
            cv2.rectangle(canvas, (x + iw, y + ih), (x + w - iw, y + h - ih), 255, -1)
            # Center black dot
            cw, ch = (x + w // 2, y + h // 2)
            cv2.circle(canvas, (cw, ch), max(1, min(iw, ih) // 2), 0, -1)

        # Draw outer borders
        cv2.rectangle(canvas, (40, 120), (self.canonical_width - 40, self.canonical_height - 100), 0, 2)

        # Draw sample bubbles from template
        bubble_rois = self.template.get("bubble_rois", {})
        for q_rois in bubble_rois.values():
            for opt_roi in q_rois.values():
                cx = int(opt_roi["x"] + opt_roi["w"] / 2)
                cy = int(opt_roi["y"] + opt_roi["h"] / 2)
                r = int(min(opt_roi["w"], opt_roi["h"]) / 2)
                cv2.circle(canvas, (cx, cy), r, 0, 1)

        return canvas

    def load_image(self, image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Load image from path, bytes, or numpy array and ensure portrait orientation."""
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Input image not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"OpenCV could not decode image: {path}")
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from byte buffer")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Auto-orient: if landscape, rotate 90 degrees clockwise to portrait
        h, w = img.shape[:2]
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        return img

    def align_by_features(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], int, float]:
        """Align image to reference sheet using SIFT feature matching and RANSAC homography."""
        if self._ref_des is None or len(self._ref_des) == 0:
            return None, 0, 0.0

        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= 2400:
            scale = 1.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            scale = 2000.0 / float(max_dim)
            small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        kp_img, des_img = self.sift.detectAndCompute(gray, None)
        if des_img is None or len(des_img) < 15:
            return None, 0, 0.0

        matches = self.matcher.knnMatch(des_img, self._ref_des, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < 0.80 * pair[1].distance:
                good_matches.append(pair[0])

        if len(good_matches) < 12:
            return None, len(good_matches), 0.0

        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale
        dst_pts = np.float32([self._ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            return None, len(good_matches), 0.0

        inliers = int(np.sum(mask))
        inlier_ratio = inliers / float(len(good_matches))

        if inliers >= 20:
            warped = cv2.warpPerspective(
                img, H, (self.canonical_width, self.canonical_height), flags=cv2.INTER_LINEAR
            )
            return warped, inliers, inlier_ratio

        return None, inliers, inlier_ratio

    def find_page_quadrilateral(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect the 4-corner paper sheet boundary using contour analysis."""
        h, w = img.shape[:2]
        scale = 800.0 / max(h, w)
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        img_area = small.shape[0] * small.shape[1]
        for c in contours[:8]:
            area = cv2.contourArea(c)
            if area < 0.30 * img_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                ordered = order_quad_points(pts)
                width_top = np.linalg.norm(ordered[1] - ordered[0])
                width_bot = np.linalg.norm(ordered[2] - ordered[3])
                height_left = np.linalg.norm(ordered[3] - ordered[0])
                height_right = np.linalg.norm(ordered[2] - ordered[1])

                avg_w = (width_top + width_bot) / 2.0
                avg_h = (height_left + height_right) / 2.0
                if avg_w == 0:
                    continue
                aspect = avg_h / avg_w
                if 1.15 <= aspect <= 1.85:
                    return ordered / scale

        return None

    def warp_quadrilateral(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Apply perspective warp from 4 ordered quad points to canonical dimensions."""
        dst_pts = np.array([
            [0, 0],
            [self.canonical_width - 1, 0],
            [self.canonical_width - 1, self.canonical_height - 1],
            [0, self.canonical_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(quad.astype("float32"), dst_pts)
        return cv2.warpPerspective(
            img, M, (self.canonical_width, self.canonical_height), flags=cv2.INTER_LINEAR
        )

    def evaluate_anchor_correlation(self, norm_gray: np.ndarray) -> Dict[str, float]:
        """Compute template matching correlation scores for each anchor."""
        scores = {}
        search_margin = 40
        for name, tpl in self._anchor_templates.items():
            if tpl.std() < 2.0:
                scores[name] = 0.0
                continue

            roi = self.anchors[name]
            y1 = max(0, roi["y"] - search_margin)
            y2 = min(self.canonical_height, roi["y"] + roi["h"] + search_margin)
            x1 = max(0, roi["x"] - search_margin)
            x2 = min(self.canonical_width, roi["x"] + roi["w"] + search_margin)

            search_region = norm_gray[y1:y2, x1:x2]
            if search_region.shape[0] < tpl.shape[0] or search_region.shape[1] < tpl.shape[1]:
                scores[name] = 0.0
                continue

            if search_region.std() < 2.0:
                scores[name] = 0.0
                continue

            res = cv2.matchTemplate(search_region, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            scores[name] = float(np.clip(max_val, 0.0, 1.0))

        return scores

    def deskew(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate and correct residual skew angle via Hough line consensus."""
        h, w = gray.shape[:2]
        center_crop = gray[int(h * 0.25):int(h * 0.75), int(w * 0.15):int(w * 0.85)]

        edges = cv2.Canny(center_crop, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

        if lines is None or len(lines) < 8:
            return gray, 0.0

        angles = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) < 15.0:
                angles.append(angle)

        if len(angles) < 5:
            return gray, 0.0

        median_angle = float(np.median(angles))
        std_angle = float(np.std(angles))

        if abs(median_angle) < 1.0 or std_angle > 1.8:
            return gray, 0.0

        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            gray, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed, median_angle

    def normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """Remove uneven shadows and normalize lighting via background division and CLAHE."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        norm = np.float32(gray) / (np.float32(bg) + 1e-5) * 255.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        return clahe.apply(norm)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Denoise while preserving sharp bubble contours."""
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian thresholding (inverted: marks = 255, background = 0)."""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )

    def compute_alignment_confidence(
        self,
        anchor_scores: Dict[str, float],
        inlier_ratio: float = 1.0,
        inliers: int = 0,
        method: str = "direct"
    ) -> float:
        """Calculate alignment quality score in [0.0, 1.0]."""
        mean_anchor = (
            sum(anchor_scores.values()) / float(len(anchor_scores))
            if anchor_scores else 0.50
        )

        if method == "direct_scan":
            method_weight = 0.95
        elif method == "sift_features":
            # 50+ inliers gives full feature confidence
            inlier_conf = np.clip(inliers / 60.0, 0.50, 1.0)
            method_weight = 0.50 * inlier_conf + 0.50 * inlier_ratio
        elif method == "contour+anchors":
            method_weight = 0.85
        else:
            method_weight = 0.40

        total_confidence = 0.60 * mean_anchor + 0.40 * method_weight
        return float(np.clip(total_confidence, 0.0, 1.0))

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        force_method: Optional[str] = None
    ) -> PreprocessResult:
        """Execute the full preprocessing pipeline on any input image."""
        raw_bgr = self.load_image(image_input)
        h, w = raw_bgr.shape[:2]

        flags = []
        method_used = "unknown"
        warped_bgr: Optional[np.ndarray] = None
        inlier_ratio = 1.0
        inliers = 0

        # 1. Perspective alignment: SIFT feature matching is primary across all sheets
        # (including flatbed scans where scanner margins otherwise cause horizontal/vertical ROI drift)
        if force_method == "direct_scan":
            warped_bgr = cv2.resize(raw_bgr, (self.canonical_width, self.canonical_height))
            method_used = "direct_scan"
        elif force_method != "contour":
            sift_warped, inliers, ratio = self.align_by_features(raw_bgr)
            if sift_warped is not None:
                warped_bgr = sift_warped
                inlier_ratio = ratio
                method_used = "sift_features"

        if warped_bgr is None and force_method != "sift_features":
            # Fallback 1: page contour quadrilateral detection
            quad = self.find_page_quadrilateral(raw_bgr)
            if quad is not None:
                warped_bgr = self.warp_quadrilateral(raw_bgr, quad)
                method_used = "contour+anchors"

        if warped_bgr is None:
            # Fallback 2: direct resize if SIFT and contour both failed
            warped_bgr = cv2.resize(raw_bgr, (self.canonical_width, self.canonical_height))
            method_used = "resize_fallback"
            flags.append("ALIGNMENT_FAILED")

        # 2. Illumination Normalization (background division + CLAHE)
        canonical_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        normalized_gray = self.normalize_illumination(canonical_gray)

        # 3. Anchor Correlation Evaluation
        anchor_scores = self.evaluate_anchor_correlation(normalized_gray)

        # 4. Deskew check (only applied if residual skew is significant)
        deskewed_norm, skew_angle = self.deskew(normalized_gray)

        # 5. Denoise & Threshold
        denoised_gray = self.denoise(deskewed_norm)
        thresholded = self.adaptive_threshold(denoised_gray)

        # 6. Compute alignment confidence score
        confidence = self.compute_alignment_confidence(
            anchor_scores, inlier_ratio, inliers, method_used
        )

        if confidence < self.confidence_threshold:
            flags.append("LOW_ALIGNMENT_CONFIDENCE")

        return PreprocessResult(
            canonical_bgr=warped_bgr,
            canonical_gray=canonical_gray,
            normalized_gray=deskewed_norm,
            thresholded=thresholded,
            confidence=round(confidence, 4),
            flags=flags,
            method_used=method_used,
            metadata={
                "original_size": (w, h),
                "skew_angle": round(skew_angle, 2),
                "inliers": inliers,
                "inlier_ratio": round(inlier_ratio, 3),
                "anchor_scores": {k: round(v, 3) for k, v in anchor_scores.items()},
            }
        )
