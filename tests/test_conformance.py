"""The adapter `flow_grpo_wan` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "flow_grpo_wan"


if __name__ == "__main__":
    unittest.main()
