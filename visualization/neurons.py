from __future__ import annotations

import html


def probability_network_svg(
    probabilities: dict[str, float],
    predicted: str | None = None,
) -> str:
    """Decorative 3-layer schematic. Output nodes follow class probabilities, not real CNN activations."""
    classes = list(probabilities.items())
    if not classes:
        raise ValueError("probabilities is empty")

    width, height = 720, 260
    input_xs = [48] * 6
    hidden_xs = [300] * 8
    input_ys = _spread(6, height)
    hidden_ys = _spread(8, height)
    out_n = len(classes)
    out_xs = [560] * out_n
    out_ys = _spread(out_n, height)

    lines: list[str] = []
    for x1, y1 in zip(input_xs, input_ys):
        for x2, y2 in zip(hidden_xs, hidden_ys):
            lines.append(_line(x1, y1, x2, y2, "rgba(120, 200, 255, 0.12)"))
    for x1, y1 in zip(hidden_xs, hidden_ys):
        for (name, prob), x2, y2 in zip(classes, out_xs, out_ys):
            alpha = 0.10 + 0.35 * float(prob)
            color = "rgba(80, 255, 210, 0.55)" if name == predicted else f"rgba(120, 200, 255, {alpha:.2f})"
            lines.append(_line(x1, y1, x2, y2, color))

    nodes: list[str] = []
    for x, y in zip(input_xs, input_ys):
        nodes.append(_node(x, y, 7, "#7ec8ff", glow=False))
    for x, y in zip(hidden_xs, hidden_ys):
        nodes.append(_node(x, y, 8, "#9aa7ff", glow=False))
    for (name, prob), x, y in zip(classes, out_xs, out_ys):
        active = name == predicted
        radius = 8 + 8 * float(prob)
        fill = "#5cffd0" if active else "#7ec8ff"
        nodes.append(_node(x, y, radius, fill, glow=active))
        label = html.escape(str(name))
        nodes.append(
            f'<text x="{x + 22}" y="{y + 4}" fill="#d7e7ff" font-size="13" '
            f'font-family="Segoe UI, sans-serif">{label} {float(prob) * 100:.0f}%</text>'
        )

    return _wrap(width, height, "".join(lines + nodes))


def idle_network_svg() -> str:
    dummy = {"class_a": 0.0, "class_b": 0.0, "class_c": 0.0}
    return probability_network_svg(dummy, predicted=None)


def _spread(count: int, height: int, pad: int = 28) -> list[float]:
    if count == 1:
        return [height / 2]
    span = height - 2 * pad
    return [pad + span * i / (count - 1) for i in range(count)]


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def _node(x: float, y: float, radius: float, fill: str, *, glow: bool) -> str:
    extra = ' filter="url(#glow)"' if glow else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"{extra}/>'


def _wrap(width: int, height: int, inner: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="CNN schematic">'
        "<defs>"
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3.5" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
        "</defs>"
        f'<rect width="{width}" height="{height}" rx="18" fill="#0b1020"/>'
        '<text x="48" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">INPUT</text>'
        '<text x="300" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">FEATURES</text>'
        '<text x="560" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">CLASSES</text>'
        f"{inner}"
        "</svg>"
    )
