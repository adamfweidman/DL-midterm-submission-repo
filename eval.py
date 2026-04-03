from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass

try:
    import cairosvg
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - exercised in the canonical runtime env
    cairosvg = None
    np = None
    Image = None


SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", flags=re.IGNORECASE)
NUMERIC_LITERAL_REGEX = re.compile(r"(?<![#A-Za-z])[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
URL_REFERENCE_REGEX = re.compile(r"url\(\s*#.*?\)", flags=re.IGNORECASE)
URL_FUNCTION_REGEX = re.compile(r"url\((.*?)\)", flags=re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")
PATH_COMMANDS = set("MmZzLlHhVvCcSsQqTtAa")
ALLOWED_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "defs",
    "use",
    "symbol",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "stop",
    "text",
    "tspan",
    "title",
    "desc",
    "style",
    "pattern",
    "marker",
    "filter",
}
IGNORED_ATTRIBUTES = {"id", "class", "version"}


def extract_svg(text: str) -> str:
    match = SVG_REGEX.search(text)
    return match.group(0).strip() if match else text.strip()


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _require_scoring_dependencies() -> None:
    if cairosvg is None or np is None or Image is None:
        raise RuntimeError(
            "SVG scoring dependencies are missing. Install the canonical runtime requirements and use the project venv."
        )


def _contains_external_url(text: str) -> bool:
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or "data:" in lowered:
        return True
    for match in URL_FUNCTION_REGEX.finditer(text):
        target = match.group(1).strip().strip("'\"")
        if not target.startswith("#"):
            return True
    return False


def _parse_svg_root(svg_text: str) -> ET.Element | None:
    try:
        return ET.fromstring(svg_text.strip())
    except ET.ParseError:
        return None


@dataclass
class SvgValidation:
    valid: bool
    reason: str
    char_length: int
    path_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ProxyRowScore:
    row_id: str
    row_score: float | None
    skipped: bool
    skip_reason: str | None
    fallback_used: bool
    validation: SvgValidation
    reference_svg_chars: int
    predicted_svg_chars: int
    reference_render_ok: bool
    reference_render_error: str | None
    prediction_render_ok: bool
    prediction_render_error: str | None
    blank_like: bool
    ssim_rgb: float | None
    edge_f1: float | None
    visual: float | None
    tag_path_f1: float | None
    attr_signature_f1: float | None
    structural: float | None
    compactness: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation"] = self.validation.to_dict()
        return payload


@dataclass
class ProxyScoreSummary:
    mean_proxy_score: float
    row_count: int
    scored_rows: int
    skipped_reference_rows: int
    validity_rate: float
    fallback_rate: float
    blank_like_rate: float
    reference_render_failures: int
    prediction_render_failures: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_svg(svg_text: str, max_length: int = 16000, max_paths: int = 256) -> SvgValidation:
    raw = svg_text.strip()
    if not raw:
        return SvgValidation(False, "empty", 0, 0)
    if not raw.lower().startswith("<svg"):
        return SvgValidation(False, "missing_svg_root_prefix", len(raw), 0)
    if len(raw) > max_length:
        return SvgValidation(False, "too_long", len(raw), 0)

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return SvgValidation(False, f"xml_parse_error:{exc}", len(raw), 0)

    if _strip_namespace(root.tag) != "svg":
        return SvgValidation(False, "root_not_svg", len(raw), 0)
    if root.attrib.get("viewBox") != "0 0 256 256":
        return SvgValidation(False, "bad_viewbox", len(raw), 0)

    path_count = 0
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag == "script":
            return SvgValidation(False, "script_tag", len(raw), path_count)
        if tag not in ALLOWED_TAGS:
            return SvgValidation(False, f"disallowed_tag:{tag}", len(raw), path_count)
        if tag == "path":
            path_count += 1
        if path_count > max_paths:
            return SvgValidation(False, "too_many_paths", len(raw), path_count)

        text_fields = [elem.text or "", elem.tail or ""]
        for attr_name, attr_value in elem.attrib.items():
            attr_name_l = attr_name.lower()
            attr_value_str = str(attr_value)
            attr_value_l = attr_value_str.lower()
            if attr_name_l.startswith("on"):
                return SvgValidation(False, f"event_handler:{attr_name}", len(raw), path_count)
            if attr_name_l in {"href", "{http://www.w3.org/1999/xlink}href", "xlink:href"} and not attr_value_l.startswith("#"):
                return SvgValidation(False, "external_reference", len(raw), path_count)
            text_fields.append(attr_value_str)

        if any(_contains_external_url(text) for text in text_fields):
            return SvgValidation(False, "external_reference", len(raw), path_count)

    return SvgValidation(True, "ok", len(raw), path_count)


def validity_rate(results: list[SvgValidation]) -> float:
    if not results:
        return 0.0
    return sum(1 for item in results if item.valid) / len(results)


def render_svg_to_rgb(svg_text: str, size: int = 256):
    _require_scoring_dependencies()
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(background, rgba).convert("RGB")
            array = np.asarray(composited, dtype=np.float32) / 255.0
        return array, None
    except Exception as exc:  # pragma: no cover - depends on renderer and SVG content
        return None, f"{type(exc).__name__}: {exc}"


def _box_filter(image, kernel_size: int = 7):
    pad = kernel_size // 2
    if image.ndim == 2:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
        integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
        window_sum = (
            integral[kernel_size:, kernel_size:]
            - integral[:-kernel_size, kernel_size:]
            - integral[kernel_size:, :-kernel_size]
            + integral[:-kernel_size, :-kernel_size]
        )
        return window_sum / float(kernel_size * kernel_size)

    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0), (0, 0)), mode="constant")
    window_sum = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    return window_sum / float(kernel_size * kernel_size)


def ssim_rgb(reference_rgb, prediction_rgb, kernel_size: int = 7) -> float:
    _require_scoring_dependencies()
    reference_rgb = reference_rgb.astype(np.float32)
    prediction_rgb = prediction_rgb.astype(np.float32)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_ref = _box_filter(reference_rgb, kernel_size=kernel_size)
    mu_pred = _box_filter(prediction_rgb, kernel_size=kernel_size)
    sigma_ref = _box_filter(reference_rgb * reference_rgb, kernel_size=kernel_size) - (mu_ref * mu_ref)
    sigma_pred = _box_filter(prediction_rgb * prediction_rgb, kernel_size=kernel_size) - (mu_pred * mu_pred)
    sigma_cross = _box_filter(reference_rgb * prediction_rgb, kernel_size=kernel_size) - (mu_ref * mu_pred)

    numerator = (2.0 * mu_ref * mu_pred + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_ref * mu_ref + mu_pred * mu_pred + c1) * (sigma_ref + sigma_pred + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-8)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _grayscale(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _edge_mask(rgb):
    gray = _grayscale(rgb)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]
    gradient = np.sqrt(gx * gx + gy * gy)
    threshold = max(0.08, float(np.percentile(gradient, 92)) * 0.5)
    return gradient >= threshold


def _dilate_mask(mask, radius: int = 1):
    current = mask.astype(bool)
    for _ in range(radius):
        height, width = current.shape
        padded = np.pad(current, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neighborhoods = [padded[row : row + height, col : col + width] for row in range(3) for col in range(3)]
        current = np.logical_or.reduce(neighborhoods)
    return current


def edge_f1(reference_rgb, prediction_rgb, tolerance: int = 1) -> float:
    _require_scoring_dependencies()
    reference_edges = _edge_mask(reference_rgb)
    prediction_edges = _edge_mask(prediction_rgb)
    reference_count = int(reference_edges.sum())
    prediction_count = int(prediction_edges.sum())

    if reference_count == 0 and prediction_count == 0:
        return 1.0
    if reference_count == 0 or prediction_count == 0:
        return 0.0

    reference_dilated = _dilate_mask(reference_edges, radius=tolerance)
    prediction_dilated = _dilate_mask(prediction_edges, radius=tolerance)
    true_positive_prediction = int(np.logical_and(prediction_edges, reference_dilated).sum())
    true_positive_reference = int(np.logical_and(reference_edges, prediction_dilated).sum())
    precision = true_positive_prediction / prediction_count
    recall = true_positive_reference / reference_count
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _collect_tag_paths(root: ET.Element) -> Counter[str]:
    counts: Counter[str] = Counter()

    def walk(elem: ET.Element, prefix: tuple[str, ...]) -> None:
        tag = _strip_namespace(elem.tag)
        current = prefix + (tag,)
        counts["/".join(current)] += 1
        for child in elem:
            walk(child, current)

    walk(root, ())
    return counts


def _normalize_attribute_value(name: str, value: str) -> str:
    normalized = URL_REFERENCE_REGEX.sub("url(#REF)", WHITESPACE_REGEX.sub(" ", value.strip()))
    if name == "d":
        return "".join(char.upper() for char in normalized if char in PATH_COMMANDS)
    return NUMERIC_LITERAL_REGEX.sub("#", normalized)


def _collect_attr_signatures(root: ET.Element) -> Counter[str]:
    counts: Counter[str] = Counter()
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        parts = []
        for attr_name, attr_value in sorted(elem.attrib.items()):
            attr_name_stripped = _strip_namespace(attr_name)
            if attr_name_stripped in IGNORED_ATTRIBUTES or attr_name_stripped.startswith("xmlns"):
                continue
            parts.append(f"{attr_name_stripped}={_normalize_attribute_value(attr_name_stripped, str(attr_value))}")
        signature = f"{tag}|{'|'.join(parts)}" if parts else f"{tag}|<none>"
        counts[signature] += 1
    return counts


def multiset_f1(reference_counts: Counter[str], prediction_counts: Counter[str]) -> float:
    reference_total = sum(reference_counts.values())
    prediction_total = sum(prediction_counts.values())
    if reference_total == 0 and prediction_total == 0:
        return 1.0
    if reference_total == 0 or prediction_total == 0:
        return 0.0

    overlap = sum(min(reference_counts[key], prediction_counts[key]) for key in set(reference_counts) | set(prediction_counts))
    precision = overlap / prediction_total
    recall = overlap / reference_total
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def is_blank_like(rgb_image, non_white_threshold: float = 0.01) -> bool:
    _require_scoring_dependencies()
    non_white = np.any(rgb_image < 0.99, axis=2)
    return float(non_white.mean()) < non_white_threshold


def score_svg_pair(
    row_id: str,
    reference_svg: str,
    predicted_svg: str,
    *,
    fallback_used: bool = False,
) -> ProxyRowScore:
    validation = validate_svg(predicted_svg)
    reference_render, reference_render_error = render_svg_to_rgb(reference_svg)
    if reference_render is None:
        return ProxyRowScore(
            row_id=row_id,
            row_score=None,
            skipped=True,
            skip_reason=f"reference_render_error:{reference_render_error}",
            fallback_used=fallback_used,
            validation=validation,
            reference_svg_chars=len(reference_svg.strip()),
            predicted_svg_chars=len(predicted_svg.strip()),
            reference_render_ok=False,
            reference_render_error=reference_render_error,
            prediction_render_ok=False,
            prediction_render_error=None,
            blank_like=False,
            ssim_rgb=None,
            edge_f1=None,
            visual=None,
            tag_path_f1=None,
            attr_signature_f1=None,
            structural=None,
            compactness=None,
        )

    if not validation.valid:
        return ProxyRowScore(
            row_id=row_id,
            row_score=0.0,
            skipped=False,
            skip_reason=None,
            fallback_used=fallback_used,
            validation=validation,
            reference_svg_chars=len(reference_svg.strip()),
            predicted_svg_chars=len(predicted_svg.strip()),
            reference_render_ok=True,
            reference_render_error=None,
            prediction_render_ok=False,
            prediction_render_error=None,
            blank_like=False,
            ssim_rgb=None,
            edge_f1=None,
            visual=None,
            tag_path_f1=None,
            attr_signature_f1=None,
            structural=None,
            compactness=None,
        )

    prediction_render, prediction_render_error = render_svg_to_rgb(predicted_svg)
    if prediction_render is None:
        return ProxyRowScore(
            row_id=row_id,
            row_score=0.0,
            skipped=False,
            skip_reason=None,
            fallback_used=fallback_used,
            validation=validation,
            reference_svg_chars=len(reference_svg.strip()),
            predicted_svg_chars=len(predicted_svg.strip()),
            reference_render_ok=True,
            reference_render_error=None,
            prediction_render_ok=False,
            prediction_render_error=prediction_render_error,
            blank_like=False,
            ssim_rgb=None,
            edge_f1=None,
            visual=None,
            tag_path_f1=None,
            attr_signature_f1=None,
            structural=None,
            compactness=None,
        )

    reference_root = _parse_svg_root(reference_svg)
    prediction_root = _parse_svg_root(predicted_svg)
    tag_path_f1_value = 0.0
    attr_signature_f1_value = 0.0
    if reference_root is not None and prediction_root is not None:
        tag_path_f1_value = multiset_f1(_collect_tag_paths(reference_root), _collect_tag_paths(prediction_root))
        attr_signature_f1_value = multiset_f1(
            _collect_attr_signatures(reference_root),
            _collect_attr_signatures(prediction_root),
        )

    ssim_value = ssim_rgb(reference_render, prediction_render)
    edge_f1_value = edge_f1(reference_render, prediction_render)
    # Kaggle only says "SSIM + Edge F1", so the 0.6/0.4 split is an informed local proxy choice.
    visual = 0.6 * ssim_value + 0.4 * edge_f1_value
    structural = 0.6 * tag_path_f1_value + 0.4 * attr_signature_f1_value
    compactness = min(1.0, (len(reference_svg.strip()) + 50.0) / (len(predicted_svg.strip()) + 50.0))
    row_score = 0.85 * visual + 0.12 * structural + 0.03 * compactness

    return ProxyRowScore(
        row_id=row_id,
        row_score=float(np.clip(row_score, 0.0, 1.0)),
        skipped=False,
        skip_reason=None,
        fallback_used=fallback_used,
        validation=validation,
        reference_svg_chars=len(reference_svg.strip()),
        predicted_svg_chars=len(predicted_svg.strip()),
        reference_render_ok=True,
        reference_render_error=None,
        prediction_render_ok=True,
        prediction_render_error=None,
        blank_like=is_blank_like(prediction_render),
        ssim_rgb=ssim_value,
        edge_f1=edge_f1_value,
        visual=visual,
        tag_path_f1=tag_path_f1_value,
        attr_signature_f1=attr_signature_f1_value,
        structural=structural,
        compactness=compactness,
    )


def summarize_proxy_scores(rows: list[ProxyRowScore]) -> ProxyScoreSummary:
    row_count = len(rows)
    scored_rows = [row.row_score for row in rows if row.row_score is not None]
    return ProxyScoreSummary(
        mean_proxy_score=float(sum(scored_rows) / len(scored_rows)) if scored_rows else 0.0,
        row_count=row_count,
        scored_rows=len(scored_rows),
        skipped_reference_rows=sum(1 for row in rows if row.skipped),
        validity_rate=float(sum(1 for row in rows if row.validation.valid) / row_count) if row_count else 0.0,
        fallback_rate=float(sum(1 for row in rows if row.fallback_used) / row_count) if row_count else 0.0,
        blank_like_rate=float(sum(1 for row in rows if row.blank_like) / row_count) if row_count else 0.0,
        reference_render_failures=sum(1 for row in rows if row.reference_render_error is not None),
        prediction_render_failures=sum(1 for row in rows if row.prediction_render_error is not None),
    )
