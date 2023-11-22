from torch import nn
import ezkl
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-small")
model = AutoModelForCausalLM.from_pretrained("Writer/palmyra-small")

circuit = model

model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.compiled')
pk_path = os.path.join('test.pk')
vk_path = os.path.join('test.vk')
settings_path = os.path.join('settings.json')
srs_path = os.path.join('kzg.srs')
witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')
print("paths set")


circuit.to('cuda')
print("moved model to cuda")

#define a query for the little model & tokenize
text = "Why does Maksis have such strange hair?"
inputs = tokenizer(text, return_tensors="pt")
print("move to cuda")
input_ids = inputs.to('cuda')
input_ids = inputs["input_ids"]
print(input_ids.device)
print(circuit.device)
print("lets flip circuit")
# Enable inference
circuit.eval()
print("inference enabled")
    # Export the model
torch.onnx.export(circuit,               # model being run
                      input_ids,                   # model input (or a tuple for multiple inputs)
                      model_path,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

print("model exported")

data_array = ((input_ids).cpu().detach().numpy()).reshape([-1]).tolist()

data = dict(input_data = [data_array])

    # Serialize data into file:
json.dump( data, open(data_path, 'w' ))
print("data serialized")



py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # "fixed" for params means that the committed to params are used for all proofs

print("args loaded")
res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
assert res == True
print("settings generated")
res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
assert res == True
print("settings calibrated")
res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
assert res == True
print("circuit compiled")
# srs path
res = ezkl.get_srs(srs_path, settings_path)

print("about to generate witness")
res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
assert os.path.isfile(witness_path)
print("witness generated")

res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
    )
print("ezkl setup")
assert res == True
assert os.path.isfile(vk_path)
assert os.path.isfile(pk_path)
assert os.path.isfile(settings_path)

proof_path = os.path.join('test.pf')
print("about to prove")
res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        srs_path,
        "single",
    )

print(res)
assert os.path.isfile(proof_path)

# VERIFY IT
print("about to verify")
res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path,
    )

assert res == True
print("verified")
