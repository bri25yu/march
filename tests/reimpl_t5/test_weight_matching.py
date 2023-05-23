from unittest import TestCase

from tests.reimpl_t5.match_weights import *
from tests.reimpl_t5.component_test_mixins import *


class TestReimplMatchT5WeightMatching(ComponentTestMixin, TestCase):
    def test_weight_matching(self) -> None:
        num_parameters_matched = assert_property_equal(self.reimpl_model, self.t5_model, "data")

        total_parameters = self.reimpl_model.count_parameters()
        self.assertEqual(total_parameters, num_parameters_matched)
