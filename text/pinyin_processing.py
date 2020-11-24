'''
    Library used to process pinyin data
'''

import re
import pypinyin

# Rule-based Pinyin Splitting
# https://zh.wikisource.org/wiki/汉语拼音方案
def split_pinyin(pinyin):
    # End Digit
    digit = pinyin[-1]
    pinyin = pinyin[:-1]

    assert digit in ["1", "2", "3", "4", "5"]

    # Leading 
    # !!! WARNING: Longer string first, it may be prefix of shorters
    # With Fallback, "y" "w" do not exist in 汉语拼音方案
    leading_list = [
        "zh", "ch", "sh",
        
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s",
        
        # Fallback
        "y", "w"
    ]

    leading = None
    for item in leading_list:
        if pinyin.startswith(item):
            leading = item
            break

    pinyin = pinyin.lstrip(leading)

    # Trailing
    trailing_list = [
        "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
        "i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
        "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
        "v", "ve", "van", "vn",

        "er"
    ]
    trailing_subsititue = {
        "ue": "ve"
    }

    trailing = pinyin

    if trailing in trailing_subsititue:
        trailing = trailing_subsititue[trailing]
    
    assert trailing in trailing_list

    if leading:
        return [leading, trailing + digit]
    else:
        return [trailing + digit]

# Rule-based expanison of X-r "哥们儿"
# https://zh.wikisource.org/wiki/汉语拼音方案
def expand_er(text, prosody):
    result = []
    for item in prosody:
        # remove digit
        digit = item[-1]
        item_nodigit = item[:-1]

        assert digit in ["1", "2", "3", "4", "5"]

        # if ends with "r" and not "er"
        if (item_nodigit.endswith("r")) and (item_nodigit != "er"):
            result.append(item_nodigit.rstrip("r") + digit)
            result.append("er5")
        else:
            result.append(item)

    return result

# Process data
def remove_prosody_label_in_text(text):
    return re.sub(r"#[1-4]", "", text)

def process_punctations(text):
    # Filter useless punctations (parentheses)
    text = re.sub(r"[）（]", "", text)

    # Remove repeated punctations
    text = re.sub(r"，[，\s]+", "，", text)
    text = re.sub(r"…[…\s]+", "…", text)
    text = re.sub(r"…[。\s]+", "…", text)
    text = re.sub(r"—[—\s]+", "—", text)
    text = re.sub(r"。[。，\s]+", "。", text)

    # Subsitute low-frequency punctations
    text = text.replace("—", "，").replace("；", "。").replace("…", "。")

    return text

def process_line_data(text, prosody):
    # Pre-process text
    text = remove_prosody_label_in_text(text)
    text = process_punctations(text)

    # Split prosody
    prosody = prosody.split(" ")

    # Combine punctations and prosody
    punctation_map = {
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "“": "\"",
        "”": "\"",
        "：": ":",
        "、": "\\"
    }

    # Remove punctation
    text_no_punct = text
    for k in punctation_map.keys():
        text_no_punct = text_no_punct.replace(k, "")

    # Length sanity check
    # if len(text_no_punct) != len(prosody):
    #     # X-r e.g. "哥们儿" will cause length mismatch
    #     # Try to expand it using rule-based algorithm
    #     prosody = expand_er(text_no_punct, prosody)

    # Expand X-r
    prosody = expand_er(text_no_punct, prosody)

    # Length sanity check
    assert len(text_no_punct) == len(prosody)

    # Concat text and punct
    result = []
    prosody_idx = 0
    for idx in range(len(text)):
        char = text[idx]
        if char in punctation_map:
            # punctation, using subsitution
            result.append(punctation_map[char])

            # Sanity check, cannot have 2 continous punctation
            if idx > 0:
                # Rule-based check 2
                last_char = text[idx - 1]

                if not (char in ["“", "”"] or last_char in ["“", "”"]):
                    assert not (last_char in punctation_map)

        else:
            # character, using pinyin
            result += split_pinyin(prosody[prosody_idx])
            prosody_idx += 1
    
    return " ".join(result)

# Rule-based fixing of pinyin
def fix_pinyin(pinyin):
    digit = pinyin[-1]
    if not str.isdigit(digit):
        pinyin += "5"

    return pinyin

# Get prosody
def get_autolabeled_prosody(text):
    # Recursive process curly brackets
    match = re.match(r'(.*?)\{(.+?)\}(.*)', text)
    if match:
        l = get_autolabeled_prosody(match.group(1))
        m = match.group(2)
        r = get_autolabeled_prosody(match.group(3))

        return (l + " " + m + " " + r).strip()
    
    # Pre-process text
    text = remove_prosody_label_in_text(text)
    text = process_punctations(text)

    # Get pinyin (normal mode)
    pinyin = pypinyin.pinyin(text, strict=False, errors=lambda x: None, style=pypinyin.Style.TONE3)
    prosody = [fix_pinyin(item[0]) for item in pinyin]

    return " ".join(prosody)

def process_autolabeled_text(text):
    # Recursive process curly brackets
    match = re.match(r'(.*?)\{(.+?)\}(.*)', text)
    if match:
        x = match.group(2).split(" ")
        pad = "〇" * len(x)

        return process_autolabeled_text(match.group(1)) + pad + process_autolabeled_text(match.group(3))

    return text