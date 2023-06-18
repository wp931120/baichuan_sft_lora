from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch

###加载量化模型
device_map = {"": 0}
tokenizer = AutoTokenizer.from_pretrained("./baichuan-7B",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./baichuan-7B",
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                             device_map=device_map)

###组装lora
LORA_WEIGHTS = "./baichuansft/"
device = "cuda:0"
model_lora = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS
).to(device)


###进行预测
device = "cuda:0"
from transformers import  GenerationConfig
generation_config = GenerationConfig(
        temperature=0.2,
        top_p = 0.85,
        do_sample = True, 
        repetition_penalty=2.0, 
        max_new_tokens=1024,  # max_length=max_new_tokens+input_sequence

)

prompt = """
      北京有啥好玩的地方
       """
inputttext ="""###Human:\n{}###Assistant:\n:
""".format(prompt)
inputs = tokenizer(prompt,return_tensors="pt").to(device)
generate_ids = model_lora.generate(**inputs, generation_config=generation_config)
output = tokenizer.decode(generate_ids[0])
print(output)
