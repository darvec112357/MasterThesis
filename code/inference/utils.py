import re,signal

def xml_to_dict(element):
    # If the element has no children, return its text content
    if len(element) == 0:
        return element.text

    # Otherwise, create a dictionary for the element
    result = {}
    for child in element:
        child_result = xml_to_dict(child)
        if child.tag not in result:
            result[child.tag] = child_result
        else:
            # If the tag already exists, make it a list of results
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_result)
    
    # Include element attributes if they exist
    if element.attrib:
        result['@attributes'] = element.attrib

    return result

#extract prediction
def extract_pred(sample):
    res = -999
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

def remove_patterns(text):
    lines = text.split('\n')
    res = []
    for line in lines:
        if 'print(' in line or 'input(' in line:
            continue
        res.append(line)
    return '\n'.join(res).strip('\n')

def extract_pred_math_solver(pred):
    pred = pred.split('The ')[0].split('This')[0].split('These')[0]
    while pred != '' and (pred[-1] == '`' or pred[-1] == '\n' or pred[-1] == ' '):
        pred = pred[:-1]
    pred = remove_patterns(pred)
    time_limit = 1 
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
        try:
            ans = ans[0]
        except:
            ans = -999
    try:
        ans = float(ans)
    except:
        ans = -999
    return float(ans)