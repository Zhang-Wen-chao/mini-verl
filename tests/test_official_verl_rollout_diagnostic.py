import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "diagnose_qwen3_5_rollouts.py"
SPEC = importlib.util.spec_from_file_location("official_verl_rollout_diagnostic", MODULE_PATH)
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class RolloutDiagnosticTests(unittest.TestCase):
    def test_variants_isolate_qwen_thinking_and_replace_old_instruction(self):
        content = f"Solve x.\n\n{diagnostic.LEGACY_INSTRUCTION}"
        variants = diagnostic.build_variants(content)
        self.assertEqual([item[0] for item in variants], [
            "default_thinking_legacy_prompt",
            "no_thinking_legacy_prompt",
            "no_thinking_concise_boxed_prompt",
            "no_thinking_final_only_boxed_prompt",
            "no_thinking_short_solution_boxed_prompt",
        ])
        self.assertTrue(variants[0][2])
        self.assertFalse(variants[1][2])
        self.assertFalse(variants[2][2])
        self.assertFalse(variants[3][2])
        self.assertFalse(variants[4][2])
        self.assertNotIn(diagnostic.LEGACY_INSTRUCTION, variants[2][1])
        self.assertIn(diagnostic.CONCISE_INSTRUCTION, variants[2][1])
        self.assertIn(diagnostic.FINAL_ONLY_INSTRUCTION, variants[3][1])
        self.assertIn(diagnostic.SHORT_SOLUTION_INSTRUCTION, variants[4][1])

    def test_concise_input_is_not_given_the_instruction_twice(self):
        content = f"Solve x.\n\n{diagnostic.CONCISE_INSTRUCTION}"
        concise = diagnostic.build_variants(content)[2][1]
        self.assertEqual(concise.count(diagnostic.CONCISE_INSTRUCTION), 1)

    def test_final_only_input_is_not_given_the_instruction_twice(self):
        content = f"Solve x.\n\n{diagnostic.FINAL_ONLY_INSTRUCTION}"
        final_only = diagnostic.build_variants(content)[3][1]
        self.assertEqual(final_only.count(diagnostic.FINAL_ONLY_INSTRUCTION), 1)

    def test_short_solution_input_is_not_given_the_instruction_twice(self):
        content = f"Solve x.\n\n{diagnostic.SHORT_SOLUTION_INSTRUCTION}"
        short_solution = diagnostic.build_variants(content)[4][1]
        self.assertEqual(short_solution.count(diagnostic.SHORT_SOLUTION_INSTRUCTION), 1)

    def test_hf_finish_reason_distinguishes_length_and_eos(self):
        hf_path = MODULE_PATH.with_name("diagnose_qwen3_5_hf_rollouts.py")
        hf_spec = importlib.util.spec_from_file_location("official_verl_hf_rollout_diagnostic", hf_path)
        assert hf_spec and hf_spec.loader
        hf = importlib.util.module_from_spec(hf_spec)
        hf_spec.loader.exec_module(hf)
        self.assertEqual(hf.finish_reason(8, 8, 9, [1, 2]), "length")
        self.assertEqual(hf.finish_reason(2, 8, 9, [1, 9]), "stop")
        self.assertEqual(hf.finish_reason(2, 8, 9, [1, 2]), "unknown")

    def test_hf_diagnostic_exposes_concise_only_sweep_mode(self):
        hf_path = MODULE_PATH.with_name("diagnose_qwen3_5_hf_rollouts.py")
        text = hf_path.read_text(encoding="utf-8")
        self.assertIn("no_thinking_concise_boxed_prompt", text)
        self.assertIn("variant_selection", text)
        self.assertIn("by_max_tokens", text)
        self.assertIn("sample_handle.flush()", text)

    def test_vllm_diagnostic_exposes_short_solution_only_audit_mode(self):
        text = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument(\n        "--variant"', text)
        self.assertIn("no_thinking_short_solution_boxed_prompt", text)
        self.assertIn("variant_selection", text)


if __name__ == "__main__":
    unittest.main()
