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
    delay = 0
    for x1, y1 in zip(input_xs, input_ys):
        for x2, y2 in zip(hidden_xs, hidden_ys):
            delay += 1
            lines.append(_line(x1, y1, x2, y2, "rgba(120, 200, 255, 0.16)", delay, active=False))
    for x1, y1 in zip(hidden_xs, hidden_ys):
        for (name, prob), x2, y2 in zip(classes, out_xs, out_ys):
            delay += 1
            active = name == predicted
            alpha = 0.12 + 0.45 * float(prob)
            color = "rgba(80, 255, 210, 0.7)" if active else f"rgba(120, 200, 255, {alpha:.2f})"
            lines.append(_line(x1, y1, x2, y2, color, delay, active=active))

    nodes: list[str] = []
    for i, (x, y) in enumerate(zip(input_xs, input_ys)):
        nodes.append(_node(x, y, 7, "#7ec8ff", glow=False, pulse=True, delay=i * 0.12))
    for i, (x, y) in enumerate(zip(hidden_xs, hidden_ys)):
        nodes.append(_node(x, y, 8, "#9aa7ff", glow=False, pulse=True, delay=0.2 + i * 0.08))
    for (name, prob), x, y in zip(classes, out_xs, out_ys):
        active = name == predicted
        radius = 8 + 8 * float(prob)
        fill = "#5cffd0" if active else "#7ec8ff"
        nodes.append(_node(x, y, radius, fill, glow=active, pulse=True, delay=0.4, ping=active))
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


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, delay: int, *, active: bool) -> str:
    extra = " flow-hot" if active else ""
    return (
        f'<line class="flow{extra}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{2 if active else 1}" '
        f'style="animation-delay:{-0.05 * (delay % 20):.2f}s"/>'
    )


def _node(
    x: float,
    y: float,
    radius: float,
    fill: str,
    *,
    glow: bool,
    pulse: bool,
    delay: float,
    ping: bool = False,
) -> str:
    cls = []
    if glow:
        cls.append("glow")
    if pulse:
        cls.append("pulse")
    class_attr = f' class="{" ".join(cls)}"' if cls else ""
    parts = [
        f'<circle{class_attr} cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'style="animation-delay:{delay:.2f}s"/>'
    ]
    if ping:
        parts.append(
            f'<circle class="ping" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="none" stroke="#5cffd0" stroke-width="2"/>'
        )
    return "".join(parts)


def _wrap(width: int, height: int, inner: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="CNN schematic">'
        "<defs>"
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3.5" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
        "<style>"
        "@keyframes dash { to { stroke-dashoffset: -28; } }"
        "@keyframes pulse { 0%,100% { opacity: 0.55; } 50% { opacity: 1; } }"
        "@keyframes ping { 0% { opacity: 0.7; } 100% { opacity: 0; transform: scale(2.2); } }"
        ".flow { stroke-dasharray: 5 9; animation: dash 1.4s linear infinite; }"
        ".flow-hot { stroke-dasharray: 7 8; animation-duration: 0.7s; }"
        ".pulse { animation: pulse 2s ease-in-out infinite; }"
        ".glow { filter: url(#glow); }"
        ".ping { transform-box: fill-box; transform-origin: center; animation: ping 1.4s ease-out infinite; }"
        "</style>"
        "</defs>"
        f'<rect width="{width}" height="{height}" rx="18" fill="#0b1020"/>'
        '<text x="48" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">INPUT</text>'
        '<text x="300" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">FEATURES</text>'
        '<text x="560" y="18" fill="#6f87b0" font-size="11" font-family="Segoe UI, sans-serif">CLASSES</text>'
        f"{inner}"
        "</svg>"
    )
