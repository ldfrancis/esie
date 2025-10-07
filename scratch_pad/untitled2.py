from utils import *
import time

sparsity = 0.5
weights_diff = 20_000
num_levels = 8


model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


save_dir = f"logs/sparse_weights/{model_name.split('/')[-1]}"
num_samples = 128
sequence_length = 2048
num_tokens = num_samples * sequence_length
calib_data = get_fineweb_edu(num_tokens, sequence_length, tokenizer, train=True)
_, test_data = get_w2_data(num_samples, sequence_length, tokenizer)


@torch.no_grad()
def prune_press(model, calib_data, sparsity_ratio, weights_diff, num_levels, theta1=0.42, theta2=0.51, theta3=0.38, is_sparsegpt=False, device=torch.device("cuda:0"), save_dir=""):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    reconstruction_errors = []
    NUM_SAMPLES = len(calib_data)

    print("loading calibration data")
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, calib_data, device)

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer_recon_error = 0
        layer = layers[i]
        layer.to(device)
        subset = find_layers(layer)
        print(f"Pruning layer {i}/{len(layers)}")

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            n_kwargs = {}
            for k in kwargs:
                n_kwargs[k] = kwargs[k].to(dev)
            kwargs = n_kwargs

        wrapped_layers = {}
        for name in subset:
            # print(subset[name].weight.device)
            wrapped_layers[name] = WrappedGPT(subset[name], theta1=theta1, theta2=theta2, theta3=theta3, is_sparsegpt=is_sparsegpt)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                ins = inps[j].unsqueeze(0).to(device)
                # print(ins.device, next(layer.parameters()).device)
                outs[j] = layer(ins, **kwargs)[0].detach().cpu()
        for h in handles:
            h.remove()

        for name in subset:
            W = subset[name].weight.data
            min_level = min(int(sparsity // (weights_diff / W.numel())), num_levels)
            max_level = min(int((1 - sparsity) // (weights_diff / W.numel())), num_levels)
            sparsities = [sparsity + l * weights_diff / W.numel() for l in range(-min_level, max_level + 1)]

            print(f"\tPruning model.decoder.layers.{i}.{name}")
            for j, sparsity_ratio in enumerate(sparsities, start=-min_level):
                if j == max_level:
                    modify=True
                else:
                    modify=False
                
                weight = wrapped_layers[name].prune(sparsity_ratio, modify=modify)

                # save weight to disk
                if save_dir:
                    save_file = os.path.join(
                        save_dir, ("sparsegpt" if is_sparsegpt else f"standard_{theta1}_{theta2}_{theta3}"), f"model.decoder.layers.{i}.{name}",
                        f"{j}.pth"
                    )
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    torch.save(weight, save_file)
            wrapped_layers[name].clean()

        # outs_cache = outs.clone()
        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).to(device), **kwargs)[0].detach().cpu()

        # layer_recon_error = ((outs_cache.float() - outs.float())**2).sum().item()
        
        # reconstruction_errors += [layer_recon_error]
        # print(f"Layer {i} reconstruction error: {layer_recon_error}")
        inps, outs = outs, inps
        # layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 

    return reconstruction_errors

start = time.time()
prune_press(model, calib_data, sparsity, weights_diff, num_levels, theta1=0.42, theta2=0.51, theta3=0.38, is_sparsegpt=True, device=torch.device("cuda"), save_dir=save_dir)
time_taken = time.time() - start
print(f"Time taken: {time_taken/60:.2f} minutes")