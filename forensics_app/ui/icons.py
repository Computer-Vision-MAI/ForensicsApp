"""Small toolbar icons drawn with PIL, so they don't depend on the system fonts."""

from __future__ import annotations

from PIL import Image, ImageDraw


def magnifier_icon(
    sign: str | None = None,
    size: int = 18,
    color: tuple[int, int, int, int] = (45, 50, 60, 255),
) -> Image.Image:
    """Return an RGBA magnifying glass, with an optional "+" or "-" inside the lens."""
    if sign not in (None, "+", "-"):
        raise ValueError(f"Unsupported magnifier sign: {sign!r}")

    big = size * 4  # draw large, then downsample for smooth edges
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    center = big * 0.40
    radius = big * 0.30
    stroke = max(1, round(big * 0.09))
    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        outline=color,
        width=stroke,
    )
    handle_start = center + radius * 0.72
    draw.line((handle_start, handle_start, big * 0.93, big * 0.93), fill=color, width=round(big * 0.13))

    arm = radius * 0.55
    if sign is not None:
        draw.line((center - arm, center, center + arm, center), fill=color, width=stroke)
    if sign == "+":
        draw.line((center, center - arm, center, center + arm), fill=color, width=stroke)

    return image.resize((size, size), Image.Resampling.LANCZOS)
