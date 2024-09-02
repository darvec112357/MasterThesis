from vllm import LLM,SamplingParams
import random 
from datasets import load_dataset
import numpy as np
import re,json

llm = LLM(model="/cluster/project/sachan/liuron/Thesis/experiments/baseline/records/2024-07-30_16-19-50/llama/checkpoint-12626/xxx")
sampling_params = SamplingParams(temperature=0,stop=['<eos>'],max_tokens=512)
test_data = []
with open('/cluster/project/sachan/liuron/Thesis/experiments/baseline/data/test_asdiv.json','r') as json_file:
    test_data = json.load(json_file)
take = len(test_data['text'])
# take = 100
texts = test_data['text'][:take]
official_answer = test_data['correct-answer'][:take]

#extract prediction
def extract_pred(sample):
    res = ''
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    matches = re.findall(pattern, sample)
    if matches != []:
        res = float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
    return res

texts = [x.split('Response:\n')[0] + 'Response:\n' for x in texts]
# print(texts[0])
predictions = llm.generate(texts,sampling_params)
# print(predictions[0].outputs[0].text)
answers = np.array([extract_pred(output.outputs[0].text) for output in predictions])
# print(answers)
# print(official_answer)
count = 0
for i in range(take):
    if abs(answers[i] - float(official_answer[i])) < 1e-6:
        count += 1
print(count/take)