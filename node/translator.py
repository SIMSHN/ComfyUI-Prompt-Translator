import json
import re
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#tokenizer.src_lang = "zh_CN"

def translate(text,targe):
    try:
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


class PromptTextTranslation:
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

        if trans_switch == "enabled" and contains_chinese(text_trans):
            if targe_lang == "英语" :
                target_text = translate(text_trans,"en_XX")
            else:
                target_text = translate(text_trans,"zh_CN")
        else:
            target_text = text_trans
        if original_lang == "中文" :
            tokenizer.src_lang = "zh_CN"
        else:
            tokenizer.src_lang = "en_XX"
        

        print("translated: " + target_text)

        output_text = ", ".join(filter(None, [target_text, text_normal]))
        output_text = output_text.replace('，', ' ,').replace('。', ' ,').replace("  ", " ").replace(" ,", " ,").replace(",,", " ,")

        print("target: " + target_text)

        return (output_text,)
