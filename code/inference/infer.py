from vllm import LLM, SamplingParams
import pandas as pd
from datasets import load_dataset
import sys
import json
from huggingface_hub import login
from utils import *
import xml.etree.ElementTree as ET

login(token='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')

guide_prompt = """"""

def remove_duplicates(dataset):
    questions = dataset['question']
    answers = dataset['answer']
    a = {}
    for i in range(len(questions)):
        a[questions[i]] = answers[i]
    print(len(a))
    res = {'question':[],
           'answer':[]}
    for key, value in a.items():
        res['question'].append(key)
        res['answer'].append(value)
    return res

def infer(data,strategy,model_name,path,part):
    if data[:2] == 'mm':
        data_type = 'mm'
    else:
        data_type = 'gsm'
    prompt_file_name = f'/cluster/project/sachan/liuron/Thesis/experiments/inference/prompts/prompt_{data_type}_{strategy}.txt'
    with open(prompt_file_name, 'r') as file:
        guide_prompt = file.read()
    if data[:3] == 'gsm':
        dataset = load_dataset('gsm8k','main')[data[4:]]
        questions = dataset['question']
        official_answer = [extract_pred(x) for x in dataset['answer']]
    if data == 'mm':
        # dataset = load_dataset('meta-math/MetaMathQA')['train']
        # dataset = [x for x in dataset if x['type'][:3] == 'GSM']
        with open('/cluster/project/sachan/liuron/Thesis/experiments/inference/mm.json','r') as json_file:
            dataset = json.load(json_file)
        batch_size = len(dataset['answer']) // 4
        questions = dataset['question'][(part-1)*batch_size:part*batch_size]
        official_answer = [extract_pred(x) for x in dataset['answer'][(part-1)*batch_size:part*batch_size]]
    if data == 'svamp':
        with open('/cluster/project/sachan/liuron/Thesis/experiments/SVAMP.json','r') as json_file:
            dataset = json.load(json_file)
        questions = [x['Body'] + '' + x['Question'] for x in dataset]
        official_answer = [x['Answer'] for x in dataset]
    if data == 'asdiv':
        dataset_asdiv = ET.parse('/cluster/project/sachan/liuron/Thesis/experiments/ASDiv.xml')
        root = dataset_asdiv.getroot()
        xml_dict = {root.tag: xml_to_dict(root)}
        dataset = xml_dict['Machine-Reading-Corpus-File']['ProblemSet']['Problem']
        questions = [x['Body'] + '' + x['Question'] for x in dataset]
        answers = [x['Answer'].split(' ')[0] for x in dataset]
        filtered_questions = []
        filtered_answers = []
        for i in range(len(answers)):
            try:
                ans = float(answers[i])
                filtered_questions.append(questions[i])
                filtered_answers.append(answers[i])
            except:
                continue
        questions = filtered_questions
        official_answer = filtered_answers
    max_tokens = 512
    num_samples = 1
    sampling_params = SamplingParams(n=num_samples,temperature=0,top_p=1,stop=['<eos>'],max_tokens=max_tokens)
    prompts = list(map(lambda x : guide_prompt + '\n\nInput:\n' + x + '\n\nResponse:\n',questions))
    outputs = llm.generate(prompts, sampling_params)
    
    answer = [output.outputs[0].text for output in outputs]

    data_df = list(zip([output.prompt for output in outputs],answer,official_answer))
    df = pd.DataFrame(data_df,columns=['prompts','output_answer','correct_answer'])
    if data_type == 'mm':
        file_name = f'/cluster/project/sachan/liuron/Thesis/experiments/inference/{path}/{model_name}-{data}-{strategy}-part{part}.csv'
    else:
        file_name = f'/cluster/project/sachan/liuron/Thesis/experiments/inference/{path}/{model_name}-{data}-{strategy}.csv'
    df.to_csv(file_name,index=False)

if __name__ == "__main__":
    model_name = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    strategy = sys.argv[3]
    part = int(sys.argv[4])
    model_map = {'llama':'unsloth/llama-3-8b-Instruct',
             'gemma7':'unsloth/gemma-7b-it'}
    llm = LLM(model=model_map[model_name],max_model_len=4096)
    for data in ['svamp','asdiv']: #'mm' for MetaMathQA,'gsm-test'/'gsm-train' for test/train sets of gsm8k
        # for strategy in ['sub']: #'sub' for sub-questioning, 'cot' for chain of thought. Do separate runs for 'sub' and 'cot' when 'mm' is used
        print(f'Running {data}-{strategy}, part {part} with {model_name}')
        infer(data,strategy,model_name,OUTPUT_DIR,part)