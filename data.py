from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Namespace registration for clean ET serialization (no ns0: prefix)
# ---------------------------------------------------------------------------
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def default_data_dir() -> Path:
    data_dir = os.environ.get("MIDTERM_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    user = Path.home().name
    return Path(f"/scratch/{user}/midterm-project/data/kaggle/svg-generation")


def validation_ids_path(repo_root: Path) -> Path:
    return repo_root / "splits" / "validation_ids.txt"


def load_validation_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_competition_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, submission_df


def split_train_validation(train_df: pd.DataFrame, validation_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_set = set(validation_ids)
    ordered = train_df.sort_values("id").reset_index(drop=True)
    is_val = ordered["id"].astype(str).isin(validation_set)
    train_split = ordered.loc[~is_val].reset_index(drop=True)
    val_split = ordered.loc[is_val].reset_index(drop=True)
    return train_split, val_split


def subset_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    return frame.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SVG viewBox normalization
# ---------------------------------------------------------------------------

_VIEWBOX_RE = re.compile(r'viewBox="([^"]*)"')
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_CMD_LETTERS = frozenset("MmLlHhVvCcSsQqTtAaZz")

# Parameter axis map: for each uppercase command, which axis each successive
# numeric parameter belongs to.  'x' -> scale by sx, 'y' -> scale by sy,
# None -> leave unchanged (angles, flags).
_CMD_PARAM_AXES: dict[str, list[str | None]] = {
    "M": ["x", "y"],
    "L": ["x", "y"],
    "H": ["x"],
    "V": ["y"],
    "C": ["x", "y", "x", "y", "x", "y"],
    "S": ["x", "y", "x", "y"],
    "Q": ["x", "y", "x", "y"],
    "T": ["x", "y"],
    "A": ["x", "y", None, None, None, "x", "y"],  # rx ry rot flag flag x y
    "Z": [],
}

_X_ATTRS = frozenset({"x", "x1", "x2", "cx", "dx", "refX", "fx"})
_Y_ATTRS = frozenset({"y", "y1", "y2", "cy", "dy", "refY", "fy"})
_W_ATTRS = frozenset({"width", "rx", "markerWidth"})
_H_ATTRS = frozenset({"height", "ry", "markerHeight"})
_R_ATTRS = frozenset({"r"})
_UNIFORM_ATTRS = frozenset({"stroke-width", "font-size", "stroke-dashoffset"})


def _fmt(value: float) -> str:
    """Format a scaled number compactly (max 2 decimal places)."""
    if abs(value - round(value)) < 0.005:
        return str(int(round(value)))
    s = f"{value:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _scale_single_attr(text: str, factor: float) -> str:
    """Scale a single numeric attribute value, preserving units."""
    text = text.strip()
    if text.endswith("%"):
        return text
    unit = ""
    num_text = text
    for suffix in ("px", "em", "ex", "pt", "cm", "mm", "in"):
        if text.endswith(suffix):
            unit = suffix
            num_text = text[: -len(suffix)]
            break
    try:
        return _fmt(float(num_text) * factor) + unit
    except ValueError:
        return text


def _scale_path_d(d: str, sx: float, sy: float) -> str:
    """Scale coordinates inside a path ``d`` attribute."""
    result: list[str] = []
    current_cmd: str | None = None
    param_idx = 0
    pos = 0
    length = len(d)

    while pos < length:
        ch = d[pos]

        if ch in _PATH_CMD_LETTERS:
            current_cmd = ch
            param_idx = 0
            result.append(ch)
            pos += 1
            continue

        if ch in " ,\t\n\r":
            result.append(ch)
            pos += 1
            continue

        m = _NUM_RE.match(d, pos)
        if m:
            num_val = float(m.group())
            axes = _CMD_PARAM_AXES.get(current_cmd.upper() if current_cmd else "", [])
            if axes:
                axis = axes[param_idx % len(axes)]
                if axis == "x":
                    num_val *= sx
                elif axis == "y":
                    num_val *= sy
            result.append(_fmt(num_val))
            param_idx += 1
            pos = m.end()
            continue

        result.append(ch)
        pos += 1

    return "".join(result)


def _scale_points(points: str, sx: float, sy: float) -> str:
    """Scale polygon/polyline ``points`` attribute."""
    nums = _NUM_RE.findall(points)
    scaled: list[str] = []
    for i, ns in enumerate(nums):
        val = float(ns)
        scaled.append(_fmt(val * sx) if i % 2 == 0 else _fmt(val * sy))
    pairs = []
    for i in range(0, len(scaled), 2):
        if i + 1 < len(scaled):
            pairs.append(f"{scaled[i]},{scaled[i + 1]}")
        else:
            pairs.append(scaled[i])
    return " ".join(pairs)


def _scale_transform(transform: str, sx: float, sy: float) -> str:
    """Scale translation components inside a ``transform`` attribute."""

    def _translate(m: re.Match) -> str:
        nums = _NUM_RE.findall(m.group(1))
        if len(nums) >= 2:
            return f"translate({_fmt(float(nums[0]) * sx)},{_fmt(float(nums[1]) * sy)})"
        if len(nums) == 1:
            return f"translate({_fmt(float(nums[0]) * sx)})"
        return m.group(0)

    def _matrix(m: re.Match) -> str:
        nums = [float(n) for n in _NUM_RE.findall(m.group(1))]
        if len(nums) == 6:
            nums[4] *= sx
            nums[5] *= sy
            return "matrix(" + ",".join(_fmt(v) for v in nums) + ")"
        return m.group(0)

    def _rotate(m: re.Match) -> str:
        nums = _NUM_RE.findall(m.group(1))
        if len(nums) == 3:
            return f"rotate({nums[0]},{_fmt(float(nums[1]) * sx)},{_fmt(float(nums[2]) * sy)})"
        return m.group(0)

    result = re.sub(r"translate\(([^)]*)\)", _translate, transform)
    result = re.sub(r"matrix\(([^)]*)\)", _matrix, result)
    result = re.sub(r"rotate\(([^)]*)\)", _rotate, result)
    return result


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _scale_element(elem: ET.Element, sx: float, sy: float, *, is_root: bool) -> None:
    """Scale coordinate attributes on one element."""
    avg = (sx + sy) / 2.0
    for attr_name in list(elem.attrib):
        bare = _strip_ns(attr_name)
        if is_root and bare in ("width", "height", "viewBox"):
            continue
        val = elem.get(attr_name, "")
        if bare == "d":
            elem.set(attr_name, _scale_path_d(val, sx, sy))
        elif bare == "points":
            elem.set(attr_name, _scale_points(val, sx, sy))
        elif bare == "transform":
            elem.set(attr_name, _scale_transform(val, sx, sy))
        elif bare in _X_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, sx))
        elif bare in _Y_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, sy))
        elif bare in _W_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, sx))
        elif bare in _H_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, sy))
        elif bare in _R_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, avg))
        elif bare in _UNIFORM_ATTRS:
            elem.set(attr_name, _scale_single_attr(val, avg))


def normalize_svg_viewbox(svg_text: str, target_size: int = 256) -> str:
    """Normalize an SVG to a *target_size* x *target_size* coordinate space.

    Parses the existing ``viewBox``, computes per-axis scale factors, rescales
    every coordinate-bearing attribute (including ``path.d``), and rewrites the
    ``viewBox`` / ``width`` / ``height`` to the target dimensions.

    Returns the original string unchanged when the viewBox is missing, already
    at the target size, or unparseable.
    """
    svg = svg_text.strip()
    if not svg:
        return svg

    vb_match = _VIEWBOX_RE.search(svg)
    if not vb_match:
        return svg

    vb_parts = vb_match.group(1).strip().split()
    if len(vb_parts) != 4:
        return svg

    try:
        vb_w, vb_h = float(vb_parts[2]), float(vb_parts[3])
    except ValueError:
        return svg

    if vb_w <= 0 or vb_h <= 0:
        return svg

    sx = target_size / vb_w
    sy = target_size / vb_h

    # Already the right size — just ensure the string format is canonical
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return _VIEWBOX_RE.sub(
            f'viewBox="0 0 {target_size} {target_size}"', svg, count=1
        )

    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg

    root.set("viewBox", f"0 0 {target_size} {target_size}")
    for attr in ("width", "height"):
        if root.get(attr) is not None:
            root.set(attr, str(target_size))

    for elem in root.iter():
        _scale_element(elem, sx, sy, is_root=(elem is root))

    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SvgRecord:
    row_id: str
    prompt: str
    svg: str


class SvgSftDataset:
    def __init__(self, frame: pd.DataFrame, *, normalize: bool = False):
        xform = normalize_svg_viewbox if normalize else lambda s: s
        self._records = [
            SvgRecord(
                row_id=str(row["id"]),
                prompt=str(row["prompt"]),
                svg=xform(str(row["svg"])),
            )
            for _, row in frame.iterrows()
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SvgRecord:
        return self._records[index]
