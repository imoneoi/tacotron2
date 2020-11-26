'''
    Library used to process pinyin data
'''

import re
import pypinyin

import hanlp

from tqdm import tqdm

# Process text only
class TextPreprocessor:
    def __init__(self, tokenizer_batchsize=1024, tagger_batchsize=1024):
        self.tokenizer = hanlp.load('LARGE_ALBERT_BASE')
        self.tagger = hanlp.load('CTB9_POS_ALBERT_BASE')

        self.tokenizer_batchsize = tokenizer_batchsize
        self.tagger_batchsize = tagger_batchsize

    def remove_prosody_label_in_text(self, text):
        return re.sub(r"#[1-4]", "", text)

    def preprocess_punctations(self, text):
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

    def interleave(self, text, tag):
        assert len(text) == len(tag)

        result = []
        for idx in range(len(text)):
            result += ["W" + c for c in text[idx]]
            result.append("T" + tag[idx])

        return result

    def preprocess(self, batch):
        # Rule-based preprocessing
        result = []
        for text in batch:
            text = self.remove_prosody_label_in_text(text)
            text = self.preprocess_punctations(text)

            result.append(text)

        batch = result

        # Tokenization
        result = []
        bs = self.tokenizer_batchsize
        for idx in tqdm(range(0, len(batch), bs)):
            text_b = batch[idx : idx + bs]

            text_b = self.tokenizer(text_b)
            result += text_b

        batch = result

        # Tagging
        result = []
        bs = self.tokenizer_batchsize
        for idx in tqdm(range(0, len(batch), bs)):
            text_b = batch[idx : idx + bs]
            tag_b = self.tagger(text_b)
            
            # Interleave
            interleaved = [self.interleave(text, tag) for text, tag in zip(text_b, tag_b)]

            result += interleaved

        batch = result

        return batch

# Process pinyin only
class PinyinProcessor:
    PITCHES = ["1", "2", "3", "4", "5"]

    # Leading
    # !!! WARNING: Longer string first, it may be prefix of shorters
    # With Fallback, "y" "w" do not exist in 汉语拼音方案
    LEADINGS = [
        "zh", "ch", "sh",
        
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s",
        
        # Fallback
        "y", "w"
    ]

    # Trailings
    TRAILINGS = [
        "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
        "i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
        "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
        "v", "ve", "van", "vn",

        "er"
    ]
    TRAILING_SUBS = {
        "ue": "ve"
    }

    def __init__(self):
        pass

    # Rule-based Pinyin Splitting
    # https://zh.wikisource.org/wiki/汉语拼音方案
    def split_single_pinyin(self, pinyin):
        # End Digit
        digit = pinyin[-1]
        pinyin = pinyin[:-1]

        assert digit in self.PITCHES
        
        leading = None
        for item in self.LEADINGS:
            if pinyin.startswith(item):
                leading = item
                break

        pinyin = pinyin.lstrip(leading)

        # Trailing
        trailing = pinyin

        if trailing in self.TRAILING_SUBS:
            trailing = self.TRAILING_SUBS[trailing]
        
        assert trailing in self.TRAILINGS

        if leading:
            return [leading, trailing + digit]
        else:
            return [trailing + digit]

    # Rule-based expanison of X-r "哥们儿"
    # https://zh.wikisource.org/wiki/汉语拼音方案
    def expand_er(self, sentence_pinyin):
        result = []
        for item in sentence_pinyin:
            # remove digit
            digit = item[-1]
            item_nodigit = item[:-1]

            assert digit in self.PITCHES

            # if ends with "r" and not "er"
            if (item_nodigit.endswith("r")) and (item_nodigit != "er"):
                result.append(item_nodigit.rstrip("r") + digit)
                result.append("er5")
            else:
                result.append(item)

        return result

    # Rule-based fixing of pinyin
    def fix_autolabeled_pinyin(self, pinyin):
        if pinyin:
            digit = pinyin[-1]
            if not str.isdigit(digit):
                pinyin += "5"

        return pinyin

    def autolabel_pinyin(self, text):
        splitted_text = [""]
        for item in text:
            typ = item[0]
            content = item[1:]

            if typ == "W":
                splitted_text[-1] += content
            elif typ == "T":
                splitted_text.append("")
            else:
                assert False

        result = []
        for item in splitted_text:
            pinyin = pypinyin.pinyin(item, strict=False, errors=lambda x: None, style=pypinyin.Style.TONE3)

            result += [self.fix_autolabeled_pinyin(x[0]) for x in pinyin]

        return " ".join(result)

# Process text and pinyin
class FullPreprocessor:
    PUNCTATION_MAP = {
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "“": "\"",
        "”": "\"",
        "：": ":",
        "、": "\\"
    }

    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.pinyin_preprocessor = PinyinProcessor()

    def preprocess(self, text_list, pinyin_list):
        text_list = self.text_preprocessor.preprocess(text_list)

        result = []
        for text, pinyin in zip(text_list, pinyin_list):
            # Generate pinyin if not exist
            if pinyin is None:
                pinyin = self.pinyin_preprocessor.autolabel_pinyin(text)

            # Pre-process pinyin
            pinyin = pinyin.split(" ")
            pinyin = self.pinyin_preprocessor.expand_er(pinyin)

            # Align text with pinyin
            aligned_pinyin = []
            pinyin_idx = 0
            for item in text:
                typ = item[0]
                content = item[1:]
                if typ == "T":
                    aligned_pinyin.append(typ + content)
                elif content in self.PUNCTATION_MAP:
                    aligned_pinyin.append(self.PUNCTATION_MAP[content])
                elif typ == "W":
                    assert pinyin_idx < len(pinyin)

                    single_pinyin = self.pinyin_preprocessor.split_single_pinyin(pinyin[pinyin_idx])
                    pinyin_idx += 1

                    aligned_pinyin += ["W" + item for item in single_pinyin]
                else:
                    assert False

            assert pinyin_idx == len(pinyin)

            result.append(" ".join(aligned_pinyin))

        return result

if __name__ == "__main__":
    preprocessor = FullPreprocessor()
    print(preprocessor.preprocess(
        [
            "蝼蚁#1往还#2空#1垄亩#3，骐麟#1埋没#2几春秋#4。",
            "餐后#3，一行人#2又陪#1汪小菲#3到#1另一诊所#2看#1耳鼻喉科#4。",
            "娃哈哈#1企业#2研究院#1副院长#2李言郡#3算是#2“苦主#1”之一#4。"
        ],
        [
            "lou2 yi2 wang3 huan2 kong1 long2 mu3 qi2 lin2 mai2 mo4 ji3 chun1 qiu1",
            "can1 hou4 yi4 xing2 ren2 you4 pei2 wang1 xiao3 fei1 dao4 ling4 yi4 zhen2 suo3 kan4 er3 bi2 hou2 ke1",
            None
        ]
    ))