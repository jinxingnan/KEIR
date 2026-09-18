import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from keir.balanced_training import ROLES, balanced_epoch_batches, paired_configs, train_interleaved
from keir.io import write_jsonl
from keir.inference import HFMultiAdapterRuntime
from keir.training import CompletionOnlyCollator, _tokenize_record


def make_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
                                    "Q": 3, "A": 4, "7": 5, "8": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                               pad_token="[PAD]", eos_token="[EOS]")


def test_only_completion_tokens_contribute_to_supervision():
    tokenizer = make_tokenizer()
    first = _tokenize_record({"prompt": "Q A ", "response": "7"}, tokenizer, 100)
    second = _tokenize_record({"prompt": "Q ", "response": "8"}, tokenizer, 100)
    assert first["labels"] == [-100, -100, 5, 2]
    batch = CompletionOnlyCollator(tokenizer)([first, second])
    assert batch["labels"][1].tolist() == [-100, 6, 2, -100]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0]


def test_overlong_context_is_rejected():
    with pytest.raises(ValueError, match="instead of truncating"):
        _tokenize_record({"prompt": "Q A Q A ", "response": "7"}, make_tokenizer(), 3)


def test_balanced_sampler_cycles_short_stream_at_full_update_boundaries():
    plan = list(balanced_epoch_batches({"decomposer": 2, "executor": 7}, 1, 2))
    assert [role for role, _ in plan] == list(ROLES) * 4
    for role in ROLES:
        indices = [i for r, batches in plan if r == role for batch in batches for i in batch]
        assert len(indices) == 8
        assert set(indices) == set(range(2 if role == "decomposer" else 7))


def test_distributed_sampler_partitions_global_batches():
    sizes = {role: 16 for role in ROLES}
    single = list(balanced_epoch_batches(sizes, 4, 2, seed=7))
    ranks = [list(balanced_epoch_batches(sizes, 2, 2, world_size=2, rank=r, seed=7)) for r in range(2)]
    for step, (role, batches) in enumerate(single):
        for micro, indices in enumerate(batches):
            assert indices == ranks[0][step][1][micro] + ranks[1][step][1][micro]
            assert set(ranks[0][step][1][micro]).isdisjoint(ranks[1][step][1][micro])


def test_peft_training_exports_reloadable_adapters(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("This CPU training test requires CUDA_VISIBLE_DEVICES=''")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = tmp_path / "base"
        make_tokenizer().save_pretrained(base)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=7, n_positions=32, n_embd=8, n_layer=1, n_head=1,
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., bos_token_id=2, eos_token_id=2))
        model.save_pretrained(base)
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()}
        config = {"common": {"model_name_or_path": str(base), "qlora": False, "bf16": False,
                             "epochs": 2, "max_length": 32, "gradient_accumulation_steps": 2,
                             "gradient_checkpointing": True, "lora_rank": 2, "lora_alpha": 2,
                             "lora_dropout": 0., "learning_rate": 0.01, "warmup_ratio": 0.}}
        for role, count in zip(ROLES, (1, 3)):
            data = tmp_path / (role + ".jsonl")
            write_jsonl(str(data), [{"prompt": "Q ", "response": "7" if role == ROLES[0] else "8"}] * count)
            config[role] = {"train_file": str(data), "validation_file": str(data),
                            "output_dir": str(tmp_path / role)}
        metrics = train_interleaved(config)
        assert all(metrics[role]["updates"] == 4 for role in ROLES)
        backbone = transformers.AutoModelForCausalLM.from_pretrained(base)
        loaded = peft.PeftModel.from_pretrained(backbone, tmp_path / ROLES[0] / "adapter", adapter_name=ROLES[0])
        loaded.load_adapter(tmp_path / ROLES[1] / "adapter", adapter_name=ROLES[1])
        for role in ROLES:
            loaded.set_adapter(role)
            assert torch.isfinite(loaded(input_ids=torch.tensor([[3, 5]])).logits).all()
            updates = [p for name, p in loaded.named_parameters() if "lora_B." + role in name]
            assert updates and any(torch.count_nonzero(p) for p in updates)
        restored = transformers.AutoModelForCausalLM.from_pretrained(base)
        assert all(torch.equal(parameter, frozen[name]) for name, parameter in restored.named_parameters())
        runtime = HFMultiAdapterRuntime(
            str(base), {role: str(tmp_path / role / "adapter") for role in ROLES},
            qlora=False, bf16=False, nominal_parameters=1000,
        )
        for role in ROLES:
            generated = runtime.generate(role, "Q ", max_new_tokens=2)
            assert generated.stats.prompt_tokens == 1
            assert 1 <= generated.stats.generated_tokens <= 2
            assert generated.stats.estimated_flops > 0
            assert generated.stats.nominal_parameters == 1000
            assert generated.stats.loaded_parameters == runtime.parameter_count
            assert generated.prompt_fingerprint
        with pytest.raises(FileExistsError):
            paired_configs(config)
    finally:
        torch.set_num_threads(original_threads)
