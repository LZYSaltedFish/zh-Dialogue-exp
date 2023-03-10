import colossalai
from colossalai.utils import load_checkpoint
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import logging
import os
import torch
import sys as _sys
from typing import Optional, Dict, Any, List

ckpt_dir = {
    '110M-v1': '/mnt/workspace/workgroup/lizhenyu/zh_ckpt/110M-bs4-epoch3-plain-fp32/colossal_model.pt',
    '110M-v2': '/mnt/workspace/workgroup/lizhenyu/zh_ckpt/110M-bs8-epoch5-bst-fp32/colossal_model.pt',
    '110M-v3': '/mnt/workspace/workgroup/lizhenyu/zh_ckpt/110M-bs12-epoch10-bst-fp16/colossal_model.pt'
}
config_path = '/mnt/workspace/workgroup/lizhenyu/zh-Dialogue-exp/model/config.py'
wenzhong_path = 'IDEA-CCNL/Wenzhong-GPT2-110M'
colossalai.launch_from_torch(config=config_path)


def clip_text(text, max_len):
    """
    Clip text to max length, adding ellipses.
    """
    if len(text) > max_len:
        begin_text = ' '.join(text[: math.floor(0.8 * max_len)].split(' ')[:-1])
        end_text = ' '.join(
            text[(len(text) - math.floor(0.2 * max_len)) :].split(' ')[1:]
        )
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text

def colorize(text, style):
    try:
        # if we're in ipython it's okay to use colors
        __IPYTHON__
        USE_COLORS = True
    except NameError:
        USE_COLORS = _sys.stdout.isatty()

    if not USE_COLORS:
        return text

    colorstyle = os.environ.get('PARLAI_COLORSTYLE')

    RESET = '\033[0;0m'
    if style == 'red':
        return '\033[0;31m' + text + RESET
    if style == 'yellow':
        return '\033[0;93m' + text + RESET
    if style == 'green':
        return '\033[0;32m' + text + RESET
    if style == 'blue':
        return '\033[0;34m' + text + RESET
    if style == 'brightblack':
        return '\033[0;90m' + text + RESET

    if colorstyle is None or colorstyle.lower() == 'steamroller':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY_NOBK = '\033[1m'
        LIGHT_GRAY_NOBK = '\033[0m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED_NOBK = '\033[1;31m'
        HIGHLIGHT_BLUE_NOBK = '\033[0;34m'
        if style == 'highlight':
            return HIGHLIGHT_RED_NOBK + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        elif style == 'text':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY_NOBK + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'id':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        else:
            return MAGENTA + text + RESET

    if colorstyle.lower() == 'spermwhale':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY = '\033[1;37;40m'
        LIGHT_GRAY = '\033[0;37;40m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED = '\033[1;37;41m'
        HIGHLIGHT_BLUE = '\033[1;37;44m'
        if style == 'highlight':
            return HIGHLIGHT_RED + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE + text + RESET
        elif style == 'text':
            return LIGHT_GRAY + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY + text + RESET
        elif style == 'id':
            return LIGHT_GRAY + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE + text + RESET
        else:
            return MAGENTA + text + RESET

    # No colorstyle specified/found.
    return text

def display_messages(
    msg: Dict[str, Any],
    max_len: int = 1000,
) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. add_fields provides
    a list of fields in the msgs which should be displayed if verbose is off.
    """

    def _pretty_lines(indent_space, field, value, style):
        line = '{}{} {}'.format(
            indent_space, colorize('[' + field + ']:', 'field'), colorize(value, style)
        )
        return line

    line = ''
    if msg is None:
        return None

    # Possibly indent the text (for the second speaker, if two).
    space = ''

    agent_id = msg.get('id', '[no id field]')

    # Display Text
    if msg.get('text', ''):
        value = clip_text(msg['text'], max_len)
        style = 'bold_text'
        field = agent_id
        line = _pretty_lines(
            indent_space=space, field=field, value=value, style=style
        )

    return line

class Predictor():
    def __init__(self, ckpt_dir):
        self.model = GPT2LMHeadModel.from_pretrained(wenzhong_path)
        load_checkpoint(ckpt_dir, self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.tokenizer = GPT2Tokenizer.from_pretrained(wenzhong_path)
        self.set_tokenizer()

        self.history_strings = []
        self.is_finished = False
        self.text_truncate = 512
        self.beam_size = 10
        self.delimiter = self.tokenizer.eos_token
        
        self.gen_config = GenerationConfig(min_new_tokens=10, max_new_tokens=128, num_beams=10)

        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                'highlight',
            )
        )
    
    def set_tokenizer(self):
        pad_token = '<pad>'
        bos_token = '<bos>'
        self.tokenizer.add_tokens(pad_token)
        self.tokenizer.add_tokens(bos_token)
        self.tokenizer.pad_token = pad_token
        self.tokenizer.bos_token = bos_token
        self.tokenizer.truncation_size = 'left'

    def preprocess(self):
        try:
            reply = self.get_human_reply()
        except StopIteration:
            self.finalize_episode()
            self.reset()
            return

        self.update_history(reply)
    
    def predict(self):
        reply = {'id': 'BlenderBot'}

        # move to GPU if necessary
        model_input = self.message
        model_input.pop('id')
        model_input.pop('text')
        if torch.cuda.is_available():
            temp = {}
            for key in model_input.keys():
                value = model_input[key]
                if torch.is_tensor(value):
                    temp[key] = value.to('cuda')
                else:
                    temp[key] = value
            model_input = temp

        with torch.no_grad():
            output = self.model.generate(**model_input,
                                    generation_config=self.gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True)[0]
            history_len = model_input['input_ids'].shape[-1]
            response = output[history_len-1:]
            text = self.tokenizer.decode(response).replace(self.tokenizer.eos_token, '')
            reply['text'] = text
        
        return reply

    def postprocess(self, result):
        self.update_history(result)
        print(
            display_messages(result)
        )
    
    def run(self):
        while True:
            self.preprocess()

            if self.is_finished:
                logging.info('epoch done')
                break
            if self.message is None:
                continue

            reply = self.predict()
            self.postprocess(reply)

    def get_human_reply(self):
        reply = {'id': 'safeLocalHuman'}
        reply_text = input(colorize('Enter Your Message:', 'field') + ' ')
        reply_text = reply_text.replace('\\n', '\n')

        # check for episode done
        if '[DONE]' in reply_text:
            raise StopIteration

        # set reply text
        reply['text'] = reply_text

        # check if finished
        if '[EXIT]' in reply_text:
            self.is_finished = True
            raise StopIteration
        
        return reply

    def update_history(self, msg):
        self.message = msg
        text = msg['text'] + self.tokenizer.eos_token
        self.history_strings.append(text)

        # set the 'text_vec' field in the message
        history_string = self.get_history_str()
        if history_string:
            encoded_input = self.tokenizer(history_string,
                                                      truncation=True,
                                                      max_length=self.text_truncate,
                                                      return_tensors='pt')
            self.message.update(encoded_input)
        return

    
    def get_history_str(self):
        if len(self.history_strings) > 0:
            history = ''.join(self.history_strings)
            return history
        
        return None
    
    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if not self.is_finished:
            print("\n[ Preparing new chat ... ]\n")
    
    def reset(self):
        self.message = None
        self.history_strings = []

if __name__ == '__main__':
    predictor = Predictor(ckpt_dir['110M-v3'])
    predictor.run()
