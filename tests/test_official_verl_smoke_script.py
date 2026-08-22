import subprocess
import unittest
from pathlib import Path


OFFICIAL_VERL_DIR = Path(__file__).resolve().parents[1] / "official_verl"
SCRIPT = OFFICIAL_VERL_DIR / "run_qwen3_0_6b_4gpu_smoke.sh"
LOCAL_BOOTSTRAP = OFFICIAL_VERL_DIR / "bootstrap_local_official_env.sh"
WAIT_AND_RUN = OFFICIAL_VERL_DIR / "wait_and_run_qwen3_0_6b_4gpu_smoke.sh"
CALIBRATION = OFFICIAL_VERL_DIR / "run_qwen3_5_4b_4gpu_calibration.sh"
WAIT_AND_RUN_CALIBRATION = OFFICIAL_VERL_DIR / "wait_and_run_qwen3_5_4b_4gpu_calibration.sh"
PPO_CALIBRATION = OFFICIAL_VERL_DIR / "run_qwen3_5_4b_ppo_gae_calibration.sh"
PPO_GATE = OFFICIAL_VERL_DIR / "await_ppo_gae_calibration.sh"
FAIR_COMPARISON_GATE = OFFICIAL_VERL_DIR / "await_fair_algorithm_comparison.sh"
FAIR_COMPARISON_CONTINUATION = OFFICIAL_VERL_DIR / "continue_fair_grpo_after_ppo.sh"


class OfficialVerlSmokeScriptTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax_and_official_entrypoint(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("verl.experimental.one_step_off_policy.main_ppo", text)
        self.assertIn("algorithm.adv_estimator=grpo", text)
        self.assertIn("trainer.n_gpus_per_node=2", text)
        self.assertIn("rollout.n_gpus_per_node=2", text)
        self.assertIn("preflight.py", text)
        self.assertIn("--require-lock", text)
        self.assertIn("--require-runtime", text)
        self.assertLess(text.index("--require-runtime"), text.index("mkdir -p"))
        self.assertIn("PIPESTATUS[0]", text)
        self.assertIn("logs/exit_status", text)

    def test_local_bootstrap_uses_a_rebuildable_local_fsdp_vllm_environment(self):
        subprocess.run(["bash", "-n", str(LOCAL_BOOTSTRAP)], check=True)
        text = LOCAL_BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("UV_PROJECT_ENVIRONMENT", text)
        self.assertIn("UV_CACHE_DIR", text)
        self.assertIn("uv sync --python 3.12 --extra fsdp --extra vllm", text)
        modules = '("torch", "transformers", "ray", "vllm", "verl")'
        self.assertIn(modules, text)
        runtime_check = text[text.index("for module in") :]
        self.assertNotIn('mbridge")', runtime_check)

    def test_wait_and_run_script_has_valid_bash_syntax_and_runtime_check(self):
        subprocess.run(["bash", "-n", str(WAIT_AND_RUN)], check=True)
        text = WAIT_AND_RUN.read_text(encoding="utf-8")
        self.assertIn("flock -n", text)
        self.assertIn('("torch", "transformers", "ray", "vllm", "verl")', text)

    def test_calibration_waiter_never_manipulates_foreign_processes(self):
        subprocess.run(["bash", "-n", str(WAIT_AND_RUN_CALIBRATION)], check=True)
        text = WAIT_AND_RUN_CALIBRATION.read_text(encoding="utf-8")
        self.assertIn("nvidia-smi --query-compute-apps=pid", text)
        self.assertIn("will not kill or alter foreign processes", text)
        self.assertIn("flock -n", text)
        self.assertIn("starting official GRPO calibration", text)
        self.assertNotIn("kill -", text)
        self.assertNotIn("pkill", text)

    def test_4b_calibration_is_bounded_and_persists_generations(self):
        subprocess.run(["bash", "-n", str(CALIBRATION)], check=True)
        text = CALIBRATION.read_text(encoding="utf-8")
        self.assertIn("TRAINING_STEPS=${TRAINING_STEPS:-2}", text)
        self.assertIn("trainer.total_training_steps=\"$TRAINING_STEPS\"", text)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}", text)
        self.assertIn("data.train_batch_size=\"$TRAIN_BATCH_SIZE\"", text)
        self.assertIn("PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}", text)
        self.assertIn("PPO_MINI_BATCH_SIZE must not exceed TRAIN_BATCH_SIZE", text)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=\"$PPO_MINI_BATCH_SIZE\"", text)
        self.assertIn("ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}", text)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.param_offload=\"$ACTOR_PARAM_OFFLOAD\"", text)
        self.assertIn("TRAINER_GPUS=${TRAINER_GPUS:-2}", text)
        self.assertIn("ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}", text)
        self.assertIn("ROLLOUT_TP=${ROLLOUT_TP:-2}", text)
        self.assertIn("TRAINER_GPUS + ROLLOUT_GPUS must equal the four visible GPUs", text)
        self.assertIn("ROLLOUT_GPUS must be divisible by ROLLOUT_TP", text)
        self.assertIn("TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by TRAINER_GPUS", text)
        self.assertIn("ROLLOUT_N=${ROLLOUT_N:-4}", text)
        self.assertIn("AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}", text)
        self.assertIn("TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by AGENT_NUM_WORKERS", text)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-1}", text)
        self.assertIn("TEST_FREQ=${TEST_FREQ:-1}", text)
        self.assertIn("SAVE_FREQ <= 0 || TEST_FREQ <= 0", text)
        self.assertIn("actor_rollout_ref.rollout.n=\"$ROLLOUT_N\"", text)
        self.assertIn("actor_rollout_ref.rollout.agent.num_workers=\"$AGENT_NUM_WORKERS\"", text)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=\"$ROLLOUT_TP\"", text)
        self.assertIn("trainer.n_gpus_per_node=\"$TRAINER_GPUS\"", text)
        self.assertIn("rollout.n_gpus_per_node=\"$ROLLOUT_GPUS\"", text)
        self.assertIn("MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-384}", text)
        self.assertIn("MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}", text)
        self.assertIn("data.max_response_length=\"$MAX_RESPONSE_LENGTH\"", text)
        self.assertIn("data.shuffle=False", text)
        self.assertIn("+data.apply_chat_template_kwargs.enable_thinking=false", text)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=\"$PPO_MINI_BATCH_SIZE\"", text)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.offload_policy=True", text)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.optimizer_offload=True", text)
        self.assertIn("override_optimizer_config={foreach: false}", text)
        self.assertIn("actor_rollout_ref.rollout.max_model_len=\"$MAX_MODEL_LEN\"", text)
        self.assertIn("actor_rollout_ref.rollout.max_num_seqs=32", text)
        self.assertIn("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True", text)
        self.assertIn("checkpoint_engine.update_weights_bucket_megabytes=3072", text)
        self.assertIn("MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER", text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE:-}", text)
        self.assertIn("trainer.rollout_data_dir=", text)
        self.assertIn("trainer.validation_data_dir=", text)
        self.assertIn("trainer.save_freq=\"$SAVE_FREQ\"", text)
        self.assertIn("trainer.test_freq=\"$TEST_FREQ\"", text)
        self.assertIn("nvidia-smi --query-compute-apps", text)
        self.assertIn("PYTHON_BIN", text)
        self.assertIn("COMPAT_PATH", text)
        self.assertIn("PYTHONPATH", text)
        self.assertIn("exit 3", text)

    def test_4b_ppo_calibration_requires_real_gae_critic_evidence(self):
        subprocess.run(["bash", "-n", str(PPO_CALIBRATION)], check=True)
        subprocess.run(["bash", "-n", str(PPO_GATE)], check=True)
        launcher = PPO_CALIBRATION.read_text(encoding="utf-8")
        gate = PPO_GATE.read_text(encoding="utf-8")
        self.assertIn("algorithm.adv_estimator=gae", launcher)
        self.assertIn("path:$CRITIC_MODEL_PATH", launcher)
        self.assertIn("tokenizer_path:$CRITIC_MODEL_PATH", launcher)
        self.assertIn("critic.strategy=fsdp", launcher)
        self.assertIn("~critic.model", launcher)
        self.assertIn("_target_:ppo_critic_config.FSDPCriticHFModelConfig", launcher)
        self.assertNotIn("_target_:verl.workers.config.FSDPCriticModelCfg", launcher)
        self.assertIn("fsdp_config:{_target_:verl.workers.config.FSDPEngineConfig,strategy:fsdp,param_offload:true,optimizer_offload:true,offload_policy:false}", launcher)
        self.assertIn("CRITIC_ACTIVATION_OFFLOAD=${CRITIC_ACTIVATION_OFFLOAD:-false}", launcher)
        self.assertIn("enable_activation_offload:$CRITIC_ACTIVATION_OFFLOAD", launcher)
        self.assertIn("NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}", launcher)
        self.assertIn("CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}", launcher)
        self.assertIn("TRAINING_STEPS=${TRAINING_STEPS:-1}", launcher)
        self.assertIn("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}", launcher)
        self.assertIn("ROLLOUT_N=${ROLLOUT_N:-1}", launcher)
        self.assertIn("critic/vf_loss:", gate)
        self.assertIn("critic/values/mean:", gate)
        self.assertIn("critic/advantages/mean:", gate)
        self.assertIn("global_step_1", gate)
        self.assertIn("will not alter foreign processes", gate)
        self.assertNotIn("kill -", gate)

    def test_fair_comparison_gate_freezes_shared_rollout_budget(self):
        subprocess.run(["bash", "-n", str(FAIR_COMPARISON_GATE)], check=True)
        text = FAIR_COMPARISON_GATE.read_text(encoding="utf-8")
        self.assertIn("ALGORITHM to ppo or grpo", text)
        self.assertIn("TRAIN_BATCH_SIZE=3", text)
        self.assertIn("ROLLOUT_N=4", text)
        self.assertIn("AGENT_NUM_WORKERS=12", text)
        self.assertIn("TRAINER_GPUS=3", text)
        self.assertIn("ROLLOUT_GPUS=1", text)
        self.assertIn("algorithm.norm_adv_by_std_in_grpo=true", text)
        self.assertIn("PPO_CRITIC_ACTIVATION_OFFLOAD=true", text)
        self.assertIn("CRITIC_ACTIVATION_OFFLOAD=$PPO_CRITIC_ACTIVATION_OFFLOAD", text)
        self.assertIn("critic.data_loader_seed=", text)
        self.assertIn("GPU became busy during runtime preflight", text)
        self.assertIn("FAIR_COMPARISON_LEG_FINISHED", text)
        self.assertIn("will not alter foreign processes", text)
        self.assertNotIn("kill -", text)

    def test_fair_comparison_continuation_requires_a_passing_ppo_gate(self):
        subprocess.run(["bash", "-n", str(FAIR_COMPARISON_CONTINUATION)], check=True)
        text = FAIR_COMPARISON_CONTINUATION.read_text(encoding="utf-8")
        self.assertIn("PPO_RUN_ROOT", text)
        self.assertIn("FAIR_COMPARISON_LEG_FINISHED: ppo completed", text)
        self.assertIn("ALGORITHM=grpo", text)
        self.assertIn("PPO gate failed", text)
        self.assertNotIn("kill -", text)
