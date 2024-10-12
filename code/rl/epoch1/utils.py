import re,signal

#extract prediction
def extract_pred(sample):
    res = ''
    pattern = r'[$]?[-+]?\d+(?:\.\d+)?(?:,\d+)*[$]?'
    matches = re.findall(pattern, sample)
    if matches != []:
        res = float(matches[-1].replace(",", "").replace(" ", "").replace("\n", "").replace("$", "").replace("x", ""))
    return res

def solve_mwp(completion, prefix_frn='prefix.txt'):
    with open(prefix_frn, 'r') as fr:
        prefix = fr.read()
    completion = completion.rstrip("#")
    code = f"{prefix}\n{completion}"
    try:
        locs = {}
        exec(code, locs, locs)
        answer = locs["answer"]
    except Exception as e:
        # print(e)
        answer = "[invalid]"
    return answer 

# Timeout Handler 
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def extract_pred_math_solver(pred):
    time_limit = 10 
    ans = -999
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(time_limit)
        ans = solve_mwp(pred)
        signal.alarm(0)
    except TimeoutError:
        print("Function execution timed out")
        ans = -999
    finally:
        signal.alarm(0)
    if isinstance(ans, tuple) or isinstance(ans, list):
        ans = ans[0]
    try:
        ans = float(ans)
    except:
        ans = -999
    return float(ans)