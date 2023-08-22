import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel


DictStrAny = Dict[str, Any]  # type: ignore[misc]
NDArrayF64 = npt.NDArray[np.float64]
NDArrayI32 = npt.NDArray[np.int32]
NDArrayU8 = npt.NDArray[np.uint8]

class ImageSize(BaseModel):
    """Define image size in config."""

    width: int
    height: int

class Poly2D(BaseModel):
    """Polygon or polyline 2D."""

    vertices: List[Tuple[float, float]]
    types: str
    closed: bool

def poly_to_patch(
    vertices: List[Tuple[float, float]],
    types: str,
    color: Tuple[float, float, float],
    closed: bool,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)
    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw= 0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )


def poly2ds_to_mask(fig, ax, shape: ImageSize, poly2d: List[Poly2D]) -> NDArrayU8:
    """Converting Poly2D to mask."""
    ax.clear()  # Clear any previous patches from the axis
    ax.axis("off")
    ax.set_xlim(0, shape.width)
    ax.set_ylim(0, shape.height)
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for poly in poly2d:
        ax.add_patch(
            poly_to_patch(
                poly["vertices"],
                poly["types"],
                color=(1, 1, 1),
                closed=poly["closed"],
            )
        )

    fig.canvas.draw()
    mask: NDArrayU8 = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    mask = mask.reshape((shape.height, shape.width, -1))[..., 0]
    ax.clear()  # Clear the patches from the axis
    plt.close()
    return mask