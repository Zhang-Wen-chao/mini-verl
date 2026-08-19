import copy
import unittest

from mini_verl.controller import Controller
from mini_verl.protocol import Trajectory, TrajectoryBatch
from mini_verl.reward import group_relative_advantages
from mini_verl.workers import RuleRewardWorker

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and Transformers are optional locally")
class HuggingFaceTrainerTest(unittest.TestCase):
    def _model(self, *, vocab_size: int = 16, n_positions: int = 16):
        torch.manual_seed(3)
        return GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_positions=n_positions,
                n_embd=8,
                n_layer=1,
                n_head=1,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        )

    def _batch(self):
        return group_relative_advantages(
            TrajectoryBatch(
                (
                    Trajectory(
                        prompt_token_ids=(1, 2),
                        response_token_ids=(3, 4),
                        old_logprobs=(-2.0, -2.0),
                        policy_version=2,
                        group_id="a",
                        reward=1.0,
                    ),
                    Trajectory(
                        prompt_token_ids=(1, 2),
                        response_token_ids=(5,),
                        old_logprobs=(-2.0,),
                        policy_version=2,
                        group_id="a",
                        reward=0.0,
                    ),
                )
            )
        )

    def _tokenizer(self):
        class TinyTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del text, return_tensors, add_special_tokens
                return {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.tensor([[1, 1]])}

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return " ".join(str(token) for token in token_ids)

        return TinyTokenizer()

    def test_trainer_updates_a_real_causal_lm(self):
        from mini_verl.hf import HuggingFaceTrainerWorker

        model = self._model()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        trainer = HuggingFaceTrainerWorker(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            pad_token_id=0,
        )
        metrics = trainer.train(self._batch(), learner_policy_version=2)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater(metrics["token_count"], 0)
        self.assertTrue(any(not torch.equal(previous, current) for previous, current in zip(before, model.parameters())))

    def test_reference_kl_path_runs(self):
        from mini_verl.hf import HuggingFaceTrainerWorker

        model = self._model()
        reference = copy.deepcopy(model)
        trainer = HuggingFaceTrainerWorker(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            pad_token_id=0,
            reference_model=reference,
            beta=0.1,
        )
        metrics = trainer.train(self._batch(), learner_policy_version=2)
        self.assertGreaterEqual(metrics["kl_loss"], 0.0)

    def test_length_bucketed_microbatches_match_full_batch_single_update(self):
        from mini_verl.hf import HuggingFaceTrainerWorker

        torch.manual_seed(31)
        config = GPT2Config(
            vocab_size=32, n_positions=32, n_embd=8, n_layer=1, n_head=1,
            pad_token_id=0, bos_token_id=1, eos_token_id=1,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        full_model = GPT2LMHeadModel(config)
        bucketed_model = copy.deepcopy(full_model)
        batch = TrajectoryBatch((
            Trajectory((1, 2), (3, 4), (-3.0, -3.0), policy_version=2, group_id="a", advantage=1.0),
            Trajectory((1, 2), (5,), (-3.0,), policy_version=2, group_id="a", advantage=-1.0),
            Trajectory((1, 2), (6, 7, 8, 9), (-3.0, -3.0, -3.0, -3.0), policy_version=2, group_id="b", advantage=0.5),
            Trajectory((1, 2), (10, 11, 12), (-3.0, -3.0, -3.0), policy_version=2, group_id="b", advantage=-0.5),
        ))
        full = HuggingFaceTrainerWorker(
            model=full_model, optimizer=torch.optim.SGD(full_model.parameters(), lr=0.01), pad_token_id=0
        )
        bucketed = HuggingFaceTrainerWorker(
            model=bucketed_model, optimizer=torch.optim.SGD(bucketed_model.parameters(), lr=0.01),
            pad_token_id=0, train_micro_batch_size=2, train_max_padded_tokens=12,
        )
        full_metrics = full.train(batch, learner_policy_version=2)
        bucketed_metrics = bucketed.train(batch, learner_policy_version=2)

        self.assertEqual(bucketed_metrics["train_microbatch_count"], 2.0)
        self.assertLess(bucketed_metrics["train_padded_sequence_tokens"], full_metrics["train_padded_sequence_tokens"])
        self.assertAlmostEqual(full_metrics["loss"], bucketed_metrics["loss"], places=6)
        self.assertAlmostEqual(full_metrics["policy_loss"], bucketed_metrics["policy_loss"], places=6)
        for full_parameter, bucketed_parameter in zip(full_model.parameters(), bucketed_model.parameters(), strict=True):
            self.assertTrue(torch.allclose(full_parameter, bucketed_parameter, atol=2e-7, rtol=1e-5))

    def test_length_bucketed_microbatches_match_full_batch_with_reference_kl(self):
        from mini_verl.hf import HuggingFaceTrainerWorker

        torch.manual_seed(37)
        config = GPT2Config(
            vocab_size=32, n_positions=32, n_embd=8, n_layer=1, n_head=1,
            pad_token_id=0, bos_token_id=1, eos_token_id=1,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        full_model = GPT2LMHeadModel(config)
        bucketed_model = copy.deepcopy(full_model)
        full_reference = copy.deepcopy(full_model)
        bucketed_reference = copy.deepcopy(full_model)
        with torch.no_grad():
            for parameter in full_reference.parameters():
                parameter.add_(0.01)
            for source, destination in zip(full_reference.parameters(), bucketed_reference.parameters(), strict=True):
                destination.copy_(source)
        batch = TrajectoryBatch((
            Trajectory((1, 2), (3, 4), (-3.0, -3.0), policy_version=2, group_id="a", advantage=1.0),
            Trajectory((1, 2), (5,), (-3.0,), policy_version=2, group_id="a", advantage=-1.0),
            Trajectory((1, 2), (6, 7, 8, 9), (-3.0, -3.0, -3.0, -3.0), policy_version=2, group_id="b", advantage=0.5),
            Trajectory((1, 2), (10, 11, 12), (-3.0, -3.0, -3.0), policy_version=2, group_id="b", advantage=-0.5),
        ))
        full = HuggingFaceTrainerWorker(
            model=full_model, optimizer=torch.optim.SGD(full_model.parameters(), lr=0.01),
            pad_token_id=0, reference_model=full_reference, beta=0.1,
        )
        bucketed = HuggingFaceTrainerWorker(
            model=bucketed_model, optimizer=torch.optim.SGD(bucketed_model.parameters(), lr=0.01),
            pad_token_id=0, reference_model=bucketed_reference, beta=0.1,
            train_micro_batch_size=2, train_max_padded_tokens=12,
        )
        full_metrics = full.train(batch, learner_policy_version=2)
        bucketed_metrics = bucketed.train(batch, learner_policy_version=2)

        for key in ("loss", "policy_loss", "kl_loss", "mean_ratio", "clip_fraction"):
            self.assertAlmostEqual(full_metrics[key], bucketed_metrics[key], places=6)
        for full_parameter, bucketed_parameter in zip(full_model.parameters(), bucketed_model.parameters(), strict=True):
            self.assertTrue(torch.allclose(full_parameter, bucketed_parameter, atol=2e-7, rtol=1e-5))

    def test_rollout_generates_and_records_old_policy_logprobs(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        torch.manual_seed(4)
        worker = HuggingFaceRolloutWorker(
            model=self._model(),
            tokenizer=self._tokenizer(),
            prompts=[PromptExample("irrelevant", {"expected_answer": "x"})],
            group_size=2,
            max_new_tokens=2,
            do_sample=True,
            collect_generation_timings=True,
        )
        rollout = worker.rollout(policy_version=9)
        self.assertEqual(len(rollout.trajectories), 2)
        self.assertEqual(rollout.policy_versions, frozenset({9}))
        for trajectory in rollout.trajectories:
            self.assertGreaterEqual(len(trajectory.response_token_ids), 1)
            self.assertEqual(len(trajectory.old_logprobs), len(trajectory.response_token_ids))
            self.assertTrue(torch.isfinite(torch.tensor(trajectory.old_logprobs)).all())
            self.assertEqual(trajectory.metadata["expected_answer"], "x")
        timings = worker.last_generation_timings
        self.assertIsNotNone(timings)
        self.assertGreaterEqual(timings.prefill_forward_calls, 1)
        self.assertGreaterEqual(timings.prefill_seconds, 0.0)
        self.assertGreaterEqual(timings.decode_seconds, 0.0)
        rollout_timings = worker.last_rollout_timings
        self.assertIsNotNone(rollout_timings)
        self.assertIs(rollout_timings.generation, timings)
        self.assertGreaterEqual(rollout_timings.old_logprob_forward_seconds, 0.0)

    def test_rollout_micro_batches_prompts_and_preserves_trajectory_ownership(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        torch.manual_seed(11)
        worker = HuggingFaceRolloutWorker(
            model=self._model(),
            tokenizer=self._tokenizer(),
            prompts=[PromptExample("first", {"name": "first"}), PromptExample("second", {"name": "second"})],
            group_size=2,
            max_new_tokens=2,
            rollout_batch_size=2,
        )
        rollout = worker.rollout(policy_version=3)

        self.assertEqual(len(rollout.trajectories), 4)
        self.assertEqual(rollout.policy_versions, frozenset({3}))
        self.assertEqual([trajectory.group_id for trajectory in rollout.trajectories], [
            "policy-3-prompt-0", "policy-3-prompt-0", "policy-3-prompt-1", "policy-3-prompt-1",
        ])
        self.assertEqual([trajectory.metadata["name"] for trajectory in rollout.trajectories], [
            "first", "first", "second", "second",
        ])
        self.assertTrue(all(len(item.old_logprobs) == len(item.response_token_ids) for item in rollout.trajectories))

    def test_rollout_micro_batch_left_pads_variable_length_prompts(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class VariableLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del return_tensors, add_special_tokens
                token_ids = (2, 3) if text == "short" else (4, 5, 6, 7)
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return " ".join(str(int(token)) for token in token_ids)

        torch.manual_seed(13)
        worker = HuggingFaceRolloutWorker(
            model=self._model(),
            tokenizer=VariableLengthTokenizer(),
            prompts=[PromptExample("short", {"name": "short"}), PromptExample("long", {"name": "long"})],
            group_size=2,
            max_new_tokens=2,
            rollout_batch_size=2,
        )
        rollout = worker.rollout(policy_version=6)

        self.assertEqual([item.prompt_token_ids for item in rollout.trajectories], [
            (2, 3), (2, 3), (4, 5, 6, 7), (4, 5, 6, 7),
        ])
        self.assertEqual([item.metadata["name"] for item in rollout.trajectories], [
            "short", "short", "long", "long",
        ])
        self.assertTrue(all(len(item.old_logprobs) == len(item.response_token_ids) for item in rollout.trajectories))

    def test_rollout_length_bucketing_reduces_prompt_padding_and_preserves_order(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class VariableLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            lengths = {"two": 2, "eight": 8, "three": 3, "nine": 9}

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del return_tensors, add_special_tokens
                token_ids = tuple(range(2, 2 + self.lengths[text]))
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return " ".join(str(int(token)) for token in token_ids)

        prompts = [
            PromptExample("two", {"name": "two"}),
            PromptExample("eight", {"name": "eight"}),
            PromptExample("three", {"name": "three"}),
            PromptExample("nine", {"name": "nine"}),
        ]
        tokenizer = VariableLengthTokenizer()
        torch.manual_seed(23)
        unbucketed = HuggingFaceRolloutWorker(
            model=self._model(), tokenizer=tokenizer, prompts=prompts, group_size=2,
            max_new_tokens=2, rollout_batch_size=2, collect_generation_timings=True,
        ).rollout(policy_version=8)
        torch.manual_seed(23)
        bucketed_worker = HuggingFaceRolloutWorker(
            model=self._model(), tokenizer=tokenizer, prompts=prompts, group_size=2,
            max_new_tokens=2, rollout_batch_size=2, bucket_prompts_by_length=True,
            collect_generation_timings=True,
        )
        bucketed = bucketed_worker.rollout(policy_version=8)

        expected_names = ["two", "two", "eight", "eight", "three", "three", "nine", "nine"]
        self.assertEqual([item.metadata["name"] for item in bucketed.trajectories], expected_names)
        self.assertEqual([item.metadata["name"] for item in unbucketed.trajectories], expected_names)
        self.assertEqual([item.group_id for item in bucketed.trajectories], [
            "policy-8-prompt-0", "policy-8-prompt-0", "policy-8-prompt-1", "policy-8-prompt-1",
            "policy-8-prompt-2", "policy-8-prompt-2", "policy-8-prompt-3", "policy-8-prompt-3",
        ])
        self.assertEqual([item.prompt_token_ids for item in bucketed.trajectories], [
            (2, 3), (2, 3), tuple(range(2, 10)), tuple(range(2, 10)),
            (2, 3, 4), (2, 3, 4), tuple(range(2, 11)), tuple(range(2, 11)),
        ])
        self.assertTrue(all(len(item.old_logprobs) == len(item.response_token_ids) for item in bucketed.trajectories))
        self.assertTrue(all(torch.isfinite(torch.tensor(item.old_logprobs)).all() for item in bucketed.trajectories))
        stats = bucketed_worker.last_prompt_batching_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats.batch_count, 2)
        self.assertEqual(stats.real_prompt_tokens, 44)
        self.assertEqual(stats.padded_prompt_tokens, 48)
        self.assertEqual(stats.max_batch_padded_prompt_tokens, 36)
        self.assertAlmostEqual(stats.padding_ratio, 1 / 12)

    def test_rollout_prompt_token_budget_bounds_each_expanded_generation_batch(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class VariableLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del return_tensors, add_special_tokens
                token_ids = tuple(range(2, 2 + int(text)))
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return " ".join(str(int(token)) for token in token_ids)

        worker = HuggingFaceRolloutWorker(
            model=self._model(vocab_size=32, n_positions=32), tokenizer=VariableLengthTokenizer(),
            prompts=[PromptExample(length, {"length": int(length)}) for length in ("3", "24", "4", "23", "5", "22", "6", "21")],
            group_size=2, max_new_tokens=2, rollout_batch_size=8,
            bucket_prompts_by_length=True, rollout_max_padded_prompt_tokens=180,
        )
        rollout = worker.rollout(policy_version=4)

        stats = worker.last_prompt_batching_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats.batch_count, 3)
        self.assertEqual(stats.real_prompt_tokens, 216)
        self.assertEqual(stats.padded_prompt_tokens, 318)
        self.assertEqual(stats.max_batch_padded_prompt_tokens, 168)
        self.assertLessEqual(stats.padding_ratio, 0.5)
        scheduled = worker._prompt_batches([
            (index, example, worker._encode_prompt(example))
            for index, example in enumerate(worker.prompts)
        ])
        self.assertTrue(all(
            worker.group_size * len(rows) * max(len(row[2]) for row in rows) <= 180
            for rows in scheduled
        ))
        self.assertEqual([item.metadata["length"] for item in rollout.trajectories], [
            3, 3, 24, 24, 4, 4, 23, 23, 5, 5, 22, 22, 6, 6, 21, 21,
        ])
        self.assertTrue(all(len(item.old_logprobs) == len(item.response_token_ids) for item in rollout.trajectories))

    def test_rollout_prompt_token_budget_rejects_an_oversized_single_prompt(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        worker = HuggingFaceRolloutWorker(
            model=self._model(), tokenizer=self._tokenizer(),
            prompts=[PromptExample("irrelevant", {})], group_size=2, max_new_tokens=2,
            rollout_max_padded_prompt_tokens=3,
        )
        with self.assertRaisesRegex(ValueError, "smaller than one expanded prompt"):
            worker.rollout(policy_version=0)

    def test_rollout_prompt_token_budget_rejects_oversized_prompt_after_a_valid_row(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class VariableLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del return_tensors, add_special_tokens
                token_ids = tuple(range(2, 2 + int(text)))
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return ""

        worker = HuggingFaceRolloutWorker(
            model=self._model(vocab_size=32, n_positions=32), tokenizer=VariableLengthTokenizer(),
            prompts=[PromptExample("3", {}), PromptExample("24", {})],
            group_size=2, max_new_tokens=2, rollout_batch_size=8,
            rollout_max_padded_prompt_tokens=40,
        )
        with self.assertRaisesRegex(ValueError, "smaller than one expanded prompt"):
            worker.rollout(policy_version=0)

    def test_rollout_sequence_token_budget_splits_without_a_prompt_budget(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class VariableLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del return_tensors, add_special_tokens
                token_ids = tuple(range(2, 2 + int(text)))
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del skip_special_tokens
                return ""

        worker = HuggingFaceRolloutWorker(
            model=self._model(vocab_size=32, n_positions=32), tokenizer=VariableLengthTokenizer(),
            prompts=[PromptExample(length, {"length": int(length)}) for length in ("3", "4", "5", "6")],
            group_size=2, max_new_tokens=8, rollout_batch_size=8,
            rollout_max_padded_sequence_tokens=60,
        )
        rollout = worker.rollout(policy_version=5)

        stats = worker.last_prompt_batching_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats.batch_count, 2)
        self.assertEqual(stats.max_batch_padded_prompt_tokens, 24)
        self.assertEqual(stats.max_batch_padded_sequence_tokens, 56)
        scheduled = worker._prompt_batches([
            (index, example, worker._encode_prompt(example))
            for index, example in enumerate(worker.prompts)
        ])
        self.assertTrue(all(
            worker.group_size * len(rows) * (max(len(row[2]) for row in rows) + worker.max_new_tokens) <= 60
            for rows in scheduled
        ))
        self.assertEqual([item.metadata["length"] for item in rollout.trajectories], [3, 3, 4, 4, 5, 5, 6, 6])

    def test_rollout_sequence_token_budget_rejects_a_single_oversized_request(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        worker = HuggingFaceRolloutWorker(
            model=self._model(), tokenizer=self._tokenizer(),
            prompts=[PromptExample("irrelevant", {})], group_size=2, max_new_tokens=8,
            rollout_max_padded_sequence_tokens=19,
        )
        with self.assertRaisesRegex(ValueError, "smaller than one expanded prompt/response"):
            worker.rollout(policy_version=0)

    def test_rollout_rejects_prompt_that_exceeds_model_context_before_generate(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample

        class FixedLengthTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, *, return_tensors, add_special_tokens):
                del text, return_tensors, add_special_tokens
                return {
                    "input_ids": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long),
                }

            def decode(self, token_ids, *, skip_special_tokens):
                del token_ids, skip_special_tokens
                return ""

        worker = HuggingFaceRolloutWorker(
            model=self._model(n_positions=8), tokenizer=FixedLengthTokenizer(),
            prompts=[PromptExample("context-limited", {})], group_size=2, max_new_tokens=5,
        )
        with self.assertRaisesRegex(ValueError, "requires 9 positions.*capacity is 8"):
            worker.rollout(policy_version=0)

    def test_controller_runs_hugging_face_rollout_reward_and_update(self):
        from mini_verl.hf import HuggingFaceRolloutWorker, HuggingFaceTrainerWorker, PromptExample

        torch.manual_seed(5)
        model = self._model()
        controller = Controller(
            rollout_worker=HuggingFaceRolloutWorker(
                model=model,
                tokenizer=self._tokenizer(),
                prompts=[PromptExample("irrelevant", {})],
                group_size=2,
                max_new_tokens=2,
            ),
            reward_worker=RuleRewardWorker(lambda trajectory: float(trajectory.metadata["sample_index"] == 0)),
            trainer_worker=HuggingFaceTrainerWorker(
                model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01), pad_token_id=0
            ),
        )
        result = controller.run_iteration()
        self.assertEqual(result.policy_version, 0)
        self.assertEqual(result.next_policy_version, 1)
        self.assertEqual(result.trajectory_count, 2)
        self.assertGreater(result.response_token_count, 0)
        self.assertAlmostEqual(result.mean_reward, 0.5)
        self.assertTrue(torch.isfinite(torch.tensor(result.metrics["loss"])))

    def test_prefetch_controller_uses_an_independent_hf_rollout_replica_at_one_step_lag(self):
        """Exercise policy-lag semantics with real CausalLM generation and GRPO."""
        from mini_verl.async_controller import PrefetchingController
        from mini_verl.hf import HuggingFaceRolloutWorker, HuggingFaceTrainerWorker, PromptExample
        from mini_verl.pipeline import AsyncRolloutBuffer
        from mini_verl.policy_sync import ModelPolicySynchronizer

        class RecordingRolloutWorker:
            def __init__(self, worker):
                self.worker = worker
                self.versions = []

            def rollout(self, *, policy_version: int):
                self.versions.append(policy_version)
                return self.worker.rollout(policy_version=policy_version)

        torch.manual_seed(19)
        trainer_model = self._model()
        rollout_model = copy.deepcopy(trainer_model)
        rollout = RecordingRolloutWorker(
            HuggingFaceRolloutWorker(
                model=rollout_model,
                tokenizer=self._tokenizer(),
                prompts=[PromptExample("irrelevant", {})],
                group_size=2,
                max_new_tokens=2,
            )
        )
        trainer = HuggingFaceTrainerWorker(
            model=trainer_model,
            optimizer=torch.optim.AdamW(trainer_model.parameters(), lr=0.01),
            pad_token_id=0,
        )
        with AsyncRolloutBuffer(rollout, max_policy_lag=1) as buffer:
            controller = PrefetchingController(
                rollout_buffer=buffer,
                reward_worker=RuleRewardWorker(lambda trajectory: float(trajectory.metadata["sample_index"] == 0)),
                trainer_worker=trainer,
                policy_synchronizer=ModelPolicySynchronizer(trainer_model, rollout_model),
            )
            first = controller.run_iteration()
            second = controller.run_iteration()

        self.assertEqual(rollout.versions, [0, 0, 1])
        self.assertEqual(first.policy_version, 0)
        self.assertEqual(second.policy_version, 1)
        self.assertEqual(second.metrics["rollout_policy_version"], 0.0)
        self.assertEqual(second.metrics["learner_policy_version"], 1.0)
        self.assertEqual(second.metrics["policy_lag"], 1.0)
        for trainer_value, rollout_value in zip(
            trainer_model.state_dict().values(), rollout_model.state_dict().values(), strict=True
        ):
            self.assertTrue(torch.equal(trainer_value, rollout_value))


if __name__ == "__main__":
    unittest.main()
