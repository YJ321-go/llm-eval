from transformers import AutoConfig, AutoModel
from datasets import load_dataset

import requests
proxies = {
  'http': 'http://proxy.ubisoft.org:3128',
  'https': 'http://proxy.ubisoft.org:3128',
}


model_name = 'openbmb/MiniCPM-Llama3-V-2_5'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)
print(config)
config.num_hidden_layers = 6
new_model = AutoModel(config)
new_model.save_pretrained("test/test_mmodel")


"""
model = YourCustomModel()  # 自定义的模型
pretrained_dict = torch.load("pretrained_model.pth")
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
"""

