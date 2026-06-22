import os

with open(r"C:\Q2_Sai\run_all_final.py", "r", encoding="utf-8") as f:
    code = f.read()

old_ablation = """def setup_ablation_hook(model, direction_vector, layer_idx):
    \"\"\"Hook that projects OUT the direction from all token positions at one layer.\"\"\"
    layer = get_model_layers(model)[layer_idx]
    d = direction_vector.to("cuda", dtype=torch.float16)

    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: [batch, seq, hidden]
        proj = (h @ d)  # [..., ]
        h_new = h - proj.unsqueeze(-1) * d
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return layer.register_forward_hook(hook)"""

new_ablation = """def setup_ablation_hook(model, direction_vector, layer_idx):
    \"\"\"Hook that projects OUT the direction from all token positions at one layer.\"\"\"
    layer = get_model_layers(model)[layer_idx]
    d_base = direction_vector.to("cuda")

    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: [batch, seq, hidden]
        d = d_base.to(dtype=h.dtype)
        proj = (h @ d)  # [..., ]
        h_new = h - proj.unsqueeze(-1) * d
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return layer.register_forward_hook(hook)"""

old_addition = """def setup_addition_hook(model, direction_vector, layer_idx, strength=1.0):
    \"\"\"Hook that ADDS the direction to all token positions at one layer.\"\"\"
    layer = get_model_layers(model)[layer_idx]
    d = direction_vector.to("cuda", dtype=torch.float16)

    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        h_new = h + strength * d
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return layer.register_forward_hook(hook)"""

new_addition = """def setup_addition_hook(model, direction_vector, layer_idx, strength=1.0):
    \"\"\"Hook that ADDS the direction to all token positions at one layer.\"\"\"
    layer = get_model_layers(model)[layer_idx]
    d_base = direction_vector.to("cuda")

    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        d = d_base.to(dtype=h.dtype)
        h_new = h + strength * d
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return layer.register_forward_hook(hook)"""

code = code.replace(old_ablation, new_ablation)
code = code.replace(old_addition, new_addition)

with open(r"C:\Q2_Sai\run_all_final.py", "w", encoding="utf-8") as f:
    f.write(code)
