"""Coordinate frames, 2-D affine transforms and the research transform artifact.

Conventions (research interface only; production Workflow C/D conventions are untouched):
  * Every modality has its own *frame*: image pixel (col=x, row=y, origin top-left, y down) and a
    frame-local world in um:  world = origin_um + pixel * pixel_um.   origin_um is (0, 0) unless the
    data say otherwise; when it is not known it is recorded as UNKNOWN (``origin_known=False``).
  * A transform maps *points* forward:  target_world = M @ source_world + t.
  * Image resampling into the target frame uses the *inverse* map (target pixel -> source pixel).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

UNKNOWN = "UNKNOWN"


@dataclass
class Frame:
    name: str                      # e.g. "031-6_2:HE"
    modality: str                  # "HE" | "IF"
    pixel_um: float | None         # level-0 pixel spacing; None = UNKNOWN
    shape_px: tuple[int, int] | None = None   # (height, width) of level 0; None = no image
    origin_um: tuple[float, float] = (0.0, 0.0)
    origin_known: bool = False     # the files carry no stage/crop origin -> frame-local only
    image_path: str | None = None
    pyramid_downsamples: tuple[int, ...] = ()
    axis: str = "x right, y down (image)"

    def pixel_to_world(self, px):
        self._need_spacing()
        return np.asarray(px, float) * self.pixel_um + np.asarray(self.origin_um)

    def world_to_pixel(self, w):
        self._need_spacing()
        return (np.asarray(w, float) - np.asarray(self.origin_um)) / self.pixel_um

    def extent_um(self):
        """Physical imaged extent (from image size x spacing), NOT the point-cloud bounding box."""
        if self.shape_px is None or self.pixel_um is None:
            return None
        h, w = self.shape_px
        return (w * self.pixel_um, h * self.pixel_um)

    def footprint_um(self):
        e = self.extent_um()
        if e is None:
            return None
        o = np.asarray(self.origin_um)
        return o + np.array([[0, 0], [e[0], 0], [e[0], e[1]], [0, e[1]]], float)

    def _need_spacing(self):
        if self.pixel_um is None:
            raise ValueError(f"{self.name}: pixel spacing UNKNOWN; pixel<->world conversion refused")

    def to_dict(self):
        d = asdict(self)
        d["pixel_um"] = self.pixel_um if self.pixel_um is not None else UNKNOWN
        d["origin_um"] = list(self.origin_um) if self.origin_known else f"{UNKNOWN} (frame-local (0,0) used)"
        return d


@dataclass
class Affine2D:
    M: np.ndarray
    t: np.ndarray
    source_frame: str
    target_frame: str

    def __post_init__(self):
        self.M = np.asarray(self.M, float).reshape(2, 2)
        self.t = np.asarray(self.t, float).reshape(2)

    def apply(self, p):
        return np.asarray(p, float) @ self.M.T + self.t

    def inverse(self):
        Mi = np.linalg.inv(self.M)
        return Affine2D(Mi, -Mi @ self.t, self.target_frame, self.source_frame)

    def compose(self, first: "Affine2D"):
        """self o first  (apply ``first`` then ``self``)."""
        if first.target_frame != self.source_frame:
            raise ValueError(f"frame mismatch: {first.target_frame} -> {self.source_frame}")
        return Affine2D(self.M @ first.M, self.M @ first.t + self.t, first.source_frame, self.target_frame)

    def params(self):
        det = float(np.linalg.det(self.M))
        s = float(np.sqrt(abs(det)))
        C = np.diag([1.0, np.sign(det) if det != 0 else 1.0])
        R = (self.M @ C) / max(s, 1e-12)
        sv = np.linalg.svd(self.M)[1]
        return {"scale": s, "reflect": det < 0, "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                "anisotropy": float(sv[0] / sv[1]), "det": det}

    def pixel_matrix(self, src: Frame, dst: Frame):
        """3x3 homogeneous matrix mapping source *pixels* to target *pixels*."""
        S = np.eye(3); S[:2, :2] *= src.pixel_um; S[:2, 2] = src.origin_um
        D = np.eye(3); D[:2, :2] /= dst.pixel_um; D[:2, 2] = -np.asarray(dst.origin_um) / dst.pixel_um
        A = np.eye(3); A[:2, :2] = self.M; A[:2, 2] = self.t
        return D @ A @ S

    def to_dict(self):
        return {"M": self.M.tolist(), "t": self.t.tolist(), "source_frame": self.source_frame,
                "target_frame": self.target_frame, "params": self.params()}

    @staticmethod
    def from_dict(d):
        return Affine2D(d["M"], d["t"], d["source_frame"], d["target_frame"])

    @staticmethod
    def identity(frame):
        return Affine2D(np.eye(2), np.zeros(2), frame, frame)


# ----------------------------------------------------------------------------- artifact
STATUSES = ("IMAGE_AVAILABLE", "IMAGE_USABLE", "IMAGE_INCOMPATIBLE", "IMAGE_REGISTRATION_UNCERTAIN", "POINT_ONLY_FALLBACK")
ARTIFACT_SCHEMA = "oir-transform-artifact/0.1 (research)"


@dataclass
class TransformArtifact:
    """What a Workflow-D-like consumer needs; produced by the research runner only."""
    sample: str
    method: str                       # E0 / E0x / E1 / E2 / E3 ...
    mode: str                         # P0 / P1 / P2
    source_frame: dict
    target_frame: dict
    forward_transform: dict           # source world -> target world (points)
    inverse_transform: dict           # target world -> source world (used for image resampling)
    transform_direction: str          # "computed source->target" or "inverted from computed target->source"
    output_frame: str
    model: str                        # similarity | affine
    roi: dict = field(default_factory=dict)
    validity_region_target_um: list = field(default_factory=list)   # source footprint mapped into target
    inputs: dict = field(default_factory=dict)                      # which modalities/images were used
    image_status: list = field(default_factory=list)                 # subset of STATUSES
    confidence: dict = field(default_factory=dict)
    failure_reasons: list = field(default_factory=list)
    accepted: bool = False
    resampling: dict = field(default_factory=lambda: {
        "mapping": "inverse (target pixel -> source pixel)", "interpolation": "bilinear (order 1)",
        "outside": "constant 0 (no extrapolation)", "validity": "source footprint polygon in target frame"})
    schema: str = ARTIFACT_SCHEMA

    def write(self, path):
        Path(path).write_text(json.dumps(_json_safe(asdict(self)), indent=2))


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float):
        return None if not np.isfinite(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


json_safe = _json_safe
