

# def describe_cache_object(obj: Any, *, name: str = "cache", max_depth: int = 3) -> None:
#     def rec(x: Any, prefix: str, depth: int) -> None:
#         print(f"{prefix}: type={type(x)}")

#         if isinstance(x, torch.Tensor):
#             print(
#                 f"{prefix}: tensor shape={tuple(x.shape)} "
#                 f"dtype={x.dtype} device={x.device} "
#                 f"data_ptr={x.data_ptr()}"
#             )
#             return

#         if depth <= 0:
#             return

#         for attr in (
#             "key_cache",
#             "value_cache",
#             "layers",
#             "layer_classes",
#             "cache_position",
#             "_seen_tokens",
#         ):
#             if hasattr(x, attr):
#                 try:
#                     value = getattr(x, attr)
#                 except Exception as e:
#                     print(f"{prefix}.{attr}: <error reading: {e}>")
#                     continue
#                 print(f"{prefix}.{attr}: type={type(value)}")
#                 rec(value, f"{prefix}.{attr}", depth - 1)

#         if isinstance(x, dict):
#             for k, v in list(x.items())[:4]:
#                 rec(v, f"{prefix}[{k!r}]", depth - 1)
#             return

#         if isinstance(x, (list, tuple)):
#             print(f"{prefix}: len={len(x)}")
#             for i, v in enumerate(list(x)[:4]):
#                 rec(v, f"{prefix}[{i}]", depth - 1)
#             return

#         try:
#             length = len(x)  # type: ignore[arg-type]
#             print(f"{prefix}: len={length}")
#         except Exception:
#             pass

#     rec(obj, name, max_depth)


# def clone_dynamic_cache_tensors(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
#     cloned_layers = []

#     for layer in cache.layers:
#         keys = getattr(layer, "keys", None)
#         values = getattr(layer, "values", None)

#         if keys is None:
#             keys = getattr(layer, "key_cache", None)
#         if values is None:
#             values = getattr(layer, "value_cache", None)

#         if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
#             raise TypeError(
#                 f"Could not find tensor keys/values on cache layer type {type(layer)}. "
#                 f"Inspect layer attrs with dir(layer)."
#             )

#         cloned_layers.append((keys.detach().clone(), values.detach().clone()))

#     return cloned_layers


# def assert_dynamic_cache_matches_snapshot(
#     cache: Any,
#     snapshot: list[tuple[torch.Tensor, torch.Tensor]],
# ) -> None:
#     if len(cache.layers) != len(snapshot):
#         raise AssertionError(
#             f"Layer count changed: cache has {len(cache.layers)}, "
#             f"snapshot has {len(snapshot)}."
#         )

#     for layer_idx, (layer, (expected_k, expected_v)) in enumerate(
#         zip(cache.layers, snapshot)
#     ):
#         actual_k = getattr(layer, "keys", None)
#         actual_v = getattr(layer, "values", None)

#         if actual_k is None:
#             actual_k = getattr(layer, "key_cache", None)
#         if actual_v is None:
#             actual_v = getattr(layer, "value_cache", None)

#         if not isinstance(actual_k, torch.Tensor) or not isinstance(actual_v, torch.Tensor):
#             raise TypeError(f"Could not find tensor keys/values on layer {layer_idx}.")

#         for name, actual, expected in (
#             ("key", actual_k, expected_k),
#             ("value", actual_v, expected_v),
#         ):
#             if actual.shape != expected.shape:
#                 raise AssertionError(
#                     f"Layer {layer_idx} {name} shape changed: "
#                     f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}."
#                 )

#             if actual.dtype != expected.dtype:
#                 raise AssertionError(
#                     f"Layer {layer_idx} {name} dtype changed: "
#                     f"actual={actual.dtype}, expected={expected.dtype}."
#                 )

#             if not torch.equal(actual, expected):
#                 diff = (actual - expected).abs()
#                 raise AssertionError(
#                     f"Layer {layer_idx} {name} changed after draft+crop: "
#                     f"num_diff={(actual != expected).sum().item()}, "
#                     f"max_abs_diff={diff.max().item()}."
#                 )
            


# def summarize_bridge_layernorms(bridge: nn.Module) -> None:
#     for norm_name in ("gap_norm", "prev_norm"):
#         norm = getattr(bridge, norm_name, None)

#         if not isinstance(norm, nn.LayerNorm):
#             print(f"{norm_name}: not found or not nn.LayerNorm")
#             continue

#         print(f"\n{norm_name}")
#         print(f"  requires_grad weight: {norm.weight.requires_grad}")
#         print(f"  requires_grad bias:   {norm.bias.requires_grad}")

#         for param_name, param, init_value in (
#             ("weight", norm.weight.detach().float(), 1.0),
#             ("bias", norm.bias.detach().float(), 0.0),
#         ):
#             delta = param - init_value

#             print(f"  {param_name}:")
#             print(f"    mean:       {param.mean().item(): .6f}")
#             print(f"    std:        {param.std(unbiased=False).item(): .6f}")
#             print(f"    min:        {param.min().item(): .6f}")
#             print(f"    max:        {param.max().item(): .6f}")
#             print(f"    abs_max:    {param.abs().max().item(): .6f}")
#             print(f"    delta_mean: {delta.mean().item(): .6f}")
#             print(f"    delta_std:  {delta.std(unbiased=False).item(): .6f}")
#             print(f"    delta_min:  {delta.min().item(): .6f}")
#             print(f"    delta_max:  {delta.max().item(): .6f}")

#             if param_name == "weight":
#                 near_zero = (param.abs() < 1e-3).float().mean().item()
#                 very_small = (param.abs() < 0.1).float().mean().item()
#                 large = (param.abs() > 3.0).float().mean().item()
#                 very_large = (param.abs() > 10.0).float().mean().item()

#                 print(f"    frac |w| < 1e-3: {near_zero:.4%}")
#                 print(f"    frac |w| < 0.1:  {very_small:.4%}")
#                 print(f"    frac |w| > 3:    {large:.4%}")
#                 print(f"    frac |w| > 10:   {very_large:.4%}")
