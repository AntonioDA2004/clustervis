import unittest
from clustervis.colors import compute_weighted_rgb

class TestRGB(unittest.TestCase):

    def test_compute_weighted_rgb_with_large_numbers(self):
        weights = [1000, 2000, 3000]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        expected_rgb = (42.5, 85.0, 127.5)  # Expected weighted average RGB
        result = compute_weighted_rgb(weights, colors)
        self.assertEqual(result, expected_rgb)


if __name__ == "__main__":
    unittest.main()
