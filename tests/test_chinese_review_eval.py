"""Native-label scoring boundaries: alternatives, exclusions and full-key coverage."""
import unittest

from vimeml.benchmarks.chinese_review_eval import summarize


def case(answers, baseline, chosen, decision="single", baseline_full=True, chosen_full=True):
    return {
        "annotation": {"answers": answers, "decision": decision},
        "candidates": [{"text": baseline, "answers_key": baseline_full},
                       {"text": chosen, "answers_key": chosen_full}],
        "predictions": {"rime": {"text": baseline, "full_key": baseline_full},
                        "model": {"text": chosen, "full_key": chosen_full}},
    }


class NativeReviewMetricsTests(unittest.TestCase):
    def test_multiple_answers_are_separate_from_first_listed_preference(self):
        result = summarize([case(["他来了", "她来了"], "她来了", "他来了", "multiple")])
        self.assertEqual(result["methods"]["rime"]["accepted_set_hits"], 1)
        self.assertEqual(result["methods"]["rime"]["first_listed_hits"], 0)
        self.assertEqual(result["methods"]["model"]["first_listed_hits"], 1)
        self.assertEqual(result["methods"]["model"]["rescued"], 0)

    def test_uncertain_and_invalid_are_not_reintroduced_by_nonempty_answers(self):
        result = summarize([
            case(["答案"], "答案", "答案"),
            case([], "随便", "随便", "uncertain"),
            case(["答案"], "答案", "答案", "invalid"),
        ])
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["scorable"], 1)
        self.assertEqual(result["methods"]["model"]["accepted_rate"], 1)

    def test_prefix_match_is_not_a_full_answer_and_breaks_are_retained(self):
        result = summarize([
            case(["答案"], "答案", "答案", baseline_full=False),
            case(["另一个答案"], "另一个答案", "未列出的写法"),
        ])
        self.assertEqual(result["methods"]["model"]["rescued"], 1)
        self.assertEqual(result["methods"]["model"]["broken"], 1)
        self.assertEqual(result["methods"]["rime"]["no_full_output"], 1)
        self.assertEqual(result["methods"]["model"]["full_output_unlisted"], 1)

    def test_no_implicit_punctuation_or_pronoun_normalization(self):
        result = summarize([case(["他来了"], "他来了。", "她来了")])
        self.assertEqual(result["candidate_ceiling"], 0)
        self.assertEqual(result["methods"]["rime"]["accepted_set_hits"], 0)
        self.assertEqual(result["methods"]["model"]["accepted_set_hits"], 0)


if __name__ == "__main__":
    unittest.main()
