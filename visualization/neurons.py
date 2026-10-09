from __future__ import annotations

import html


def probability_network_svg(
    probabilities: dict[str, float],
    predicted: str | None = None,
) -> str:
    """Decorative multi-layer schematic. Output nodes follow class probabilities, not real CNN activations."""
    classes = list(probabilities.items())
    if not classes:
        raise ValueError("probabilities is empty")

    width, height = 860, 340
    layers = [
        _layer_points(72, 7, height),
        _layer_points(290, 9, height),
        _layer_points(510, 9, height),
        _layer_points(720, len(classes), height),
    ]
    titles = [(72, "INPUT"), (290, "CONV"), (510, "FEATURES"), (720, "CLASSES")]

    parts: list[str] = []
    path_id = 0
    for layer_i, (left, right) in enumerate(zip(layers, layers[1:])):
        last = layer_i == len(layers) - 2
        for i, (x1, y1) in enumerate(left):
            targets = _targets(i, len(left), len(right))
            for t, j in enumerate(targets):
                x2, y2 = right[j]
                name, prob = (classes[j] if last else (None, 0.0))
                hot = bool(last and name == predicted)
                pid = f"p{path_id}"
                path_id += 1
                alpha = 0.10 + (0.55 * float(prob) if last else 0.08)
                color = "rgba(31,77,143,0.9)" if hot else f"rgba(90,86,78,{0.18 + alpha:.2f})"
                packet = t == 0 or hot
                parts.append(_synapse(x1, y1, x2, y2, pid, color, hot=hot, packet=packet, delay=(path_id % 9) * 0.18))

    for i, points in enumerate(layers):
        for j, (x, y) in enumerate(points):
            if i == 3:
                name, prob = classes[j]
                active = name == predicted
                radius = 9 + 10 * float(prob)
                fill = "#1f4d8f" if active else "#6a645a"
                parts.append(_node(x, y, radius, fill, glow=active, delay=0.15 * j, ping=active, core=True))
                label = html.escape(str(name))
                parts.append(
                    f'<text x="{x + 24}" y="{y + 4}" fill="#1c1c1c" font-size="14" '
                    f'font-family="Segoe UI, sans-serif">{label} {float(prob) * 100:.0f}%</text>'
                )
            else:
                fill = "#3f3a34" if i else "#2b2b2b"
                parts.append(_node(x, y, 8 if i else 7, fill, glow=False, delay=0.1 * j, ping=False, core=True))

    heading = "".join(
        f'<text x="{x}" y="22" fill="#5a564e" font-size="11" font-family="Segoe UI, sans-serif">{title}</text>'
        for x, title in titles
    )
    return _wrap(width, height, heading + "".join(parts))


def idle_network_svg() -> str:
    dummy = {"class_a": 0.0, "class_b": 0.0, "class_c": 0.0}
    return probability_network_svg(dummy, predicted=None)


def _layer_points(x: float, count: int, height: int, pad: int = 38) -> list[tuple[float, float]]:
    if count == 1:
        return [(x, height / 2)]
    span = height - 2 * pad
    return [(x, pad + span * i / (count - 1)) for i in range(count)]


def _targets(index: int, n_from: int, n_to: int) -> list[int]:
    center = round(index * (n_to - 1) / max(n_from - 1, 1))
    values = [(center + delta) % n_to for delta in (-1, 0, 1)]
    # Keep order stable and unique.
    seen: list[int] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _synapse(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pid: str,
    color: str,
    *,
    hot: bool,
    packet: bool,
    delay: float,
) -> str:
    extra = " flow-hot" if hot else ""
    dur = 0.9 if hot else 1.8
    path = (
        f'<path id="{pid}" class="flow{extra}" d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="{2.4 if hot else 1.1}" '
        f'style="animation-delay:{-delay:.2f}s"/>'
    )
    if not packet:
        return path
    spark_fill = "#1f4d8f" if hot else "#8a8378"
    spark = (
        f'<circle class="spark" r="{3.6 if hot else 2.6}" fill="{spark_fill}">'
        f'<animateMotion dur="{dur}s" repeatCount="indefinite" begin="{delay:.2f}s">'
        f'<mpath href="#{pid}" xlink:href="#{pid}"/>'
        "</animateMotion></circle>"
    )
    return path + spark


def _node(
    x: float,
    y: float,
    radius: float,
    fill: str,
    *,
    glow: bool,
    delay: float,
    ping: bool,
    core: bool,
) -> str:
    cls = ["pulse"]
    if glow:
        cls.append("glow")
    body = (
        f'<circle class="{" ".join(cls)}" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
        f'fill="{fill}" style="animation-delay:{delay:.2f}s"/>'
    )
    if core:
        body += (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(2.0, radius * 0.35):.1f}" '
            f'fill="#f4f1ea" opacity="0.9"/>'
        )
    if ping:
        body += (
            f'<circle class="ping" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="none" stroke="#1f4d8f" stroke-width="2"/>'
        )
        body += (
            f'<circle class="ping ping-slow" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="none" stroke="#6a645a" stroke-width="1.2"/>'
        )
    return body


def _wrap(width: int, height: int, inner: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="CNN schematic">'
        "<defs>"
        '<filter id="glow" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="4.5" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
        "<style>"
        "@keyframes dash { to { stroke-dashoffset: -36; } }"
        "@keyframes pulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }"
        "@keyframes ping { 0% { opacity: 0.75; } 100% { opacity: 0; transform: scale(2.4); } }"
        "@keyframes sparkle { 0%,100% { opacity: 0.35; } 50% { opacity: 1; } }"
        ".flow { stroke-dasharray: 4 8; animation: dash 1.2s linear infinite; }"
        ".flow-hot { stroke-dasharray: 6 7; animation-duration: 0.55s; }"
        ".pulse { animation: pulse 1.6s ease-in-out infinite; }"
        ".glow { filter: url(#glow); }"
        ".spark { filter: url(#glow); animation: sparkle 0.8s ease-in-out infinite; }"
        ".ping { transform-box: fill-box; transform-origin: center; animation: ping 1.2s ease-out infinite; }"
        ".ping-slow { animation-duration: 2s; animation-delay: 0.4s; }"
        "</style>"
        "</defs>"
        f'<rect width="{width}" height="{height}" rx="4" fill="#fffdf8" stroke="#d8d2c6"/>'
        f"{inner}"
        "</svg>"
    )
