import os
import pytest

@pytest.mark.skipif(os.getenv("X3_RUN_REAL_CHECKPOINT") != "1",
                    reason="Requires a local/cloud copy of the 3B checkpoint and explicit opt-in")
def test_real_qwen_checkpoint_mapping_parity_cache_and_generation():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from qwen_coder_x3 import X3Config, X3ForCausalLM
    from qwen_coder_x3.weight_mapping import map_base_state_dict

    model_path = os.environ.get("X3_CHECKPOINT_PATH", "Qwen/Qwen2.5-Coder-3B-Instruct")
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    input_ids = torch.tensor([[151644, 872, 198, 791, 264, 279, 13]],
                             device=next(base.parameters()).device)
    with torch.no_grad():
        base_logits = base(input_ids=input_ids, use_cache=False).logits
    base_state = base.state_dict()  # references only; no clone
    x3 = X3ForCausalLM(base, X3Config(base_model_name_or_path=model_path)).eval()
    _, report = map_base_state_dict(base_state, x3.state_dict(), strict=True)
    assert report.ok
    if bool(base.config.tie_word_embeddings):
        assert base.get_input_embeddings().weight.data_ptr() == base.get_output_embeddings().weight.data_ptr()
    with torch.no_grad():
        output = x3(input_ids=input_ids, use_cache=True)
    assert output.logits.shape[-1] == base.config.vocab_size
    assert torch.allclose(output.logits, base_logits, atol=2e-3, rtol=2e-3)
    assert output.past_key_values is not None
    with torch.no_grad():
        generated = x3.generate(input_ids=input_ids, max_new_tokens=2, do_sample=False)
    assert generated.shape[0] == 1 and generated.shape[1] >= input_ids.shape[1]
