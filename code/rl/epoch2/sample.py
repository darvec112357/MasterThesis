from vllm import LLM,SamplingParams
import json,re,sys,signal
import random
from utils import *

num_samples = 10
llm = LLM(model="/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch1/records/2024-07-07_16-33-03/llama/checkpoint-19460/xxx",
        #   seed=412053,
        #   max_model_len=2048,
        #   swap_space=6
          )
sampling_params = SamplingParams(n=num_samples,temperature=0.7,top_p=1,stop=['<eos>'],max_tokens=300)
part = sys.argv[1]
model_name = sys.argv[2]
# print(part)
with open('/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch2/data/parts/part{}.json'.format(part), 'r') as json_file:
    dataset = json.load(json_file)
take = len(dataset['correct-answer'])
block = 0
# take = 1000
# block = 10
official_answer = dataset['correct-answer'][take*block:take*(block+1)]
texts = dataset['text'][take*block:take*(block+1)]
questions = dataset['question'][take*block:take*(block+1)]

acceptable_outputs_step1 = []
acceptable_outputs_step2 = []
# print(questions[0])
def sample(inputs,index,step=1):
    second_step_questions = []
    count = 0
    distinct_appearance = 0
    correct_strategies = {'Program of Thought':0,
                'Chain of Thought':0,
                'Sub-questioning':0}
    total_strategies = {'Program of Thought':0,
                 'Chain of Thought':0,
                 'Sub-questioning':0}
    predictions = llm.generate(inputs,sampling_params)
    noise = 0
    easy_questions = 0
    for i in range(len(predictions)):   
        correct_responses = {'Program of Thought':[],
                    'Chain of Thought':[],
                    'Sub-questioning':[]} 
        incorrect_responses = {'Program of Thought':[],
                    'Chain of Thought':[],
                    'Sub-questioning':[]} 
        temp = []
        # original_strategy = '' if step == 1 else inputs[i].split('<incorrect>')[1].split(':')[0]
        original_strategy = '' if step == 1 else inputs[i].split('<incorrect>')[1].split('\n')[0]
        for j in range(num_samples):
            response_part = predictions[i].outputs[j].text
            strategy = response_part.split(':')[0]
            if strategy not in ['Program of Thought','Chain of Thought','Sub-questioning'] or strategy == original_strategy:
                noise += 1
                continue
            if strategy not in temp: temp.append(strategy)
            output_answer = response_part.replace(strategy,'')
            if output_answer[0] == ':': output_answer = output_answer[1:]
            if output_answer[0] == ' ': output_answer = output_answer[1:]
            total_strategies[strategy] += 1
            text = predictions[i].prompt + response_part
            # question = predictions[i].prompt.split('\n<incorrect>')[0].split('Input:\nQuestion: ')[1]
            # if question not in questions:
            #     print(question)
            #     exit()
            # assert(question in questions)
            ans = extract_pred(response_part) if strategy != 'Program of Thought' else extract_pred_math_solver(output_answer)
            correct_answer = official_answer[i] if step == 1 else official_answer[index[i]]
            if abs(ans - float(correct_answer)) < 1e-6:
                count += 1
                correct_responses[strategy].append(text)
                correct_strategies[strategy] += 1
            else:
                incorrect_responses[strategy].append(text)
        distinct_appearance += len(temp)       
        # if len(correct_responses['Chain of Thought']) > len(incorrect_responses['Chain of Thought']) and \
        #     len(correct_responses['Sub-questioning']) > len(incorrect_responses['Sub-questioning']) and \
        #     len(correct_responses['Program of Thought']) > len(incorrect_responses['Program of Thought']): 
        #     easy_questions += 1
        #     continue
        for k,v in correct_responses.items():
            if len(v) > len(incorrect_responses[k]):
                if step == 1: acceptable_outputs_step1.append(v[0] + '<eos>')
                if step == 2: acceptable_outputs_step2.append(v[0] + '<eos>')
            elif len(incorrect_responses[k]) > 0 and step == 1:
                # for k2,v2 in correct_responses.items():
                #     if len(v2) > len(incorrect_responses[k2]):
                second_step_text = predictions[i].prompt.replace('<incorrect>None','<incorrect>' + k)
                second_step_questions.append(second_step_text)
                index.append(i)
                        # acceptable_outputs_step2.append(v2[0] + '<eos>')
    print(f'{distinct_appearance} strategies in total')
    print(f'{(len(inputs)*num_samples)-noise} valid outputs in total, noise rate is {str(noise*100/(len(inputs)*num_samples))}%')
    for k,v in total_strategies.items():
        print(f'{k} appears {v} times, with accuracy {str(correct_strategies[k]/v)}')
    if step == 1:
        print('Output-wise accuracy is {}.'.format(str(count/(len(inputs)*num_samples))))
        print('Strategy-wise accuracy is {}.'.format(str((len(acceptable_outputs_step1) + easy_questions)/distinct_appearance)))
        print(len(acceptable_outputs_step1))
        sample(second_step_questions,index,2)
    else:
        print('Output-wise accuracy for step 2 is {}.'.format(str(count/(len(inputs)*num_samples))))
        print(len(acceptable_outputs_step2))

sample(texts,[])
# acceptable_outputs_step1 = random.sample(acceptable_outputs_step1,len(acceptable_outputs_step1)//4)
acceptable_outputs = acceptable_outputs_step1 + acceptable_outputs_step2
random.shuffle(acceptable_outputs)
split_point = int(len(acceptable_outputs) * 0.9)
dict_train = {
    'text' : acceptable_outputs[:split_point]
}
dict_val = {
    'text' : acceptable_outputs[split_point:]
}
with open(f'/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch2/data/train/{model_name}/train_{part}.json', "w") as json_file:
    json.dump(dict_train, json_file)
with open(f'/cluster/project/sachan/liuron/Thesis/experiments/rl/epoch2/data/train/{model_name}/val_{part}.json', "w") as json_file:
    json.dump(dict_val, json_file)