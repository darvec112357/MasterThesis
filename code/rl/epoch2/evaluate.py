from vllm import LLM,SamplingParams
import json,re,sys
from utils import *
import pandas as pd
import random
import numpy as np
from collections import Counter

dataset = sys.argv[1]
llm = LLM(model='/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch1/records/2024-08-29_15-40-02/llama/checkpoint-19460/xxx')
sampling_params = SamplingParams(temperature=0.1,stop=['<eos>'],max_tokens=512)
with open(f'/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch2/data/test_{dataset}.json', 'r') as json_file:
    dataset = json.load(json_file)
take = len(dataset['text'])
# take=1319
official_answer = dataset['correct-answer'][:take]
solvable = dataset['correct'][:take]
texts = dataset['text'][:take]

temperatures = []
step1_accuracies, step2_accuracies = [], []
step1_distribution = []
step2_distribution = []

count = 0
second_step_questions = []
second_step_idx = []
predictions = llm.generate(texts,sampling_params=SamplingParams(temperature=0,stop=['<eos>'],max_tokens=512))
strategies = []
strategies_step1 = []
accuracy_dict = {'Program of Thought':0,
                'Chain of Thought':0,
                'Sub-questioning':0,
                }
for i in range(len(predictions)):
    response_part = predictions[i].outputs[0].text
    strategy = response_part.split(':')[0]
    if strategy not in ['Program of Thought','Chain of Thought','Sub-questioning','Unsolvable']:
        continue
    text = predictions[i].prompt + response_part
    output_answer = response_part.replace(strategy,'')
    if output_answer[0] == ':': output_answer = output_answer[1:]
    if output_answer[0] == ' ': output_answer = output_answer[1:]
    strategies_step1.append(strategy)
    ans = extract_pred(response_part) if strategy != 'Program of Thought' else extract_pred_math_solver(output_answer)
    try:
        ans = float(ans)
    except:
        continue
    if abs(ans - float(official_answer[i])) < 1e-6 or strategy == 'Unsolvable' and not solvable[i]:
    # if strategy_correct_map[strategy][i]:
        count += 1
        accuracy_dict[strategy] += 1
        strategies.append(strategy)
    else:
        second_question = predictions[i].prompt.replace('None',strategy + ': ' + output_answer)
        # print(second_question)
        second_step_questions.append(second_question)
        second_step_idx.append(i)
print('Accuracy for step 1 is {}'.format(str(count/take)))  
frequencies = Counter(strategies_step1)
for element, freq in frequencies.items():
    print(f"Element {element} appears {freq} times, with accuracy {accuracy_dict[element]/freq}.")
# print(second_step_questions[0])
predictions = llm.generate(second_step_questions,sampling_params=SamplingParams(temperature=0.1,stop=['<eos>'],max_tokens=512))
for i in range(len(predictions)):
    response_part = predictions[i].outputs[0].text
    text = predictions[i].prompt + response_part
    strategy = response_part.split(':')[0]
    if strategy not in ['Program of Thought','Chain of Thought','Sub-questioning','Unsolvable']:
        continue
    output_answer = response_part.replace(strategy + ': ','')
    strategies.append(strategy)
    ans = extract_pred(response_part) if strategy != 'Program of Thought' else extract_pred_math_solver(output_answer)
    try:
        ans = float(ans)
    except:
        continue
    if abs(ans - float(official_answer[second_step_idx[i]])) < 1e-6 or strategy == 'Unsolvable' and not solvable[second_step_idx[i]]:
        count += 1
        accuracy_dict[strategy] += 1
print('Accuracy for step 2 is {}'.format(str(count/take)))  
frequencies = Counter(strategies)
for element, freq in frequencies.items():
    print(f"Element {element} appears {freq} times, with accuracy {accuracy_dict[element]/freq}.")



