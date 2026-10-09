import unittest

from visualization.neurons import idle_network_svg, probability_network_svg


class NeuronSvgTest(unittest.TestCase):
    def test_predicted_class_is_in_svg(self) -> None:
        svg = probability_network_svg({"cat": 0.9, "dog": 0.1}, predicted="cat")
        self.assertIn("<svg", svg)
        self.assertIn("cat 90%", svg)
        self.assertIn("dog 10%", svg)
        self.assertIn("CLASSES", svg)
        self.assertIn("@keyframes dash", svg)
        self.assertIn("class=\"flow", svg)

    def test_escapes_class_names(self) -> None:
        svg = probability_network_svg({"a<b": 1.0}, predicted="a<b")
        self.assertIn("a&lt;b", svg)
        self.assertNotIn("a<b", svg.split("CLASSES")[-1])

    def test_idle_svg_has_no_predicted_glow_requirement(self) -> None:
        svg = idle_network_svg()
        self.assertIn("<svg", svg)
        self.assertIn("INPUT", svg)

    def test_empty_probabilities_raise(self) -> None:
        with self.assertRaises(ValueError):
            probability_network_svg({})
