import json
import re
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast,MarianMTModel, MarianTokenizer

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    

def translate(text,targe,original_lang):
    try:
        if original_lang == "中文" :
            tokenizer.src_lang = "zh_CN"
        else:
            tokenizer.src_lang = "en_XX"
        
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[targe]
            
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except TencentCloudSDKException as err:
        print("文本翻译错误：" + err)
        return text


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))


def translate1(text):
    src_text = text
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return res[0]



class PromptTextTranslation:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_lang": (["英语", "中文",],{"default": "中文"}),
                "targe_lang": (["英语", "中文",],{"default": "英语"}),
                "text_trans": ("STRING", {"multiline": True, "default": "海边，日出"}),
                "text_normal": ("STRING", {"multiline": True}),
                "trans_switch": (["enabled", "disabled"],),
                
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translation"
    CATEGORY = "utils"

    def translation(self, original_lang, targe_lang, text_trans, text_normal, trans_switch, ):

        if text_trans == "undefined":
            text_trans = ""
        if text_normal == "undefined":
            text_normal = ""

        target_text = ""

        print("prompt: ", text_trans, text_normal)

        if trans_switch == "enabled" :
            if targe_lang == "英语" :
                target_text = translate(text_trans,"en_XX",original_lang)
            else:
                target_text = translate(text_trans,"zh_CN",original_lang)
        else:
            target_text = text_trans.replace(", ", ",").replace(". ", ".").replace("， ", "，").replace("。 ", "。").lower()
            
        
        

        print("translated: " + target_text,"original_lang:" + original_lang,"targe_lang:" + targe_lang)

        output_text = ", ".join(filter(None, [target_text, text_normal]))
        output_text = output_text.replace('，', ',').replace('。', '.').replace("  ", " ").replace(" ,", ",").replace(",,", ",").replace(", ", ",").replace(". ", ".")

        print("target: " + target_text)

        return (output_text,)

class PromptTextTranslation1:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_trans": ("STRING", {"multiline": True, "default": "海边，日出"}),
                "text_normal": ("STRING", {"multiline": True}),
                "trans_switch": (["enabled", "disabled"],),
                
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translation"
    CATEGORY = "utils"
    def translation(self,text_trans, text_normal, trans_switch, ):

        if text_trans == "undefined":
            text_trans = ""
        if text_normal == "undefined":
            text_normal = ""

        target_text = ""

        print("prompt: ", text_trans, text_normal)

        if trans_switch == "enabled" :
                target_text = translate1(text_trans)
        else:
            target_text = text_trans
            
        
        

        print("translated: " + target_text)

        output_text = ", ".join(filter(None, [target_text, text_normal]))
        output_text = output_text.replace('，', ',').replace('。', '.').replace("  ", " ").replace(" ,", ",").replace(",,", ",").replace(", ", ",").replace(". ", ".")

        print("target: " + target_text)

        return (output_text,)