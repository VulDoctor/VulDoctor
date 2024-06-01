import argparse
import json
import os
import torch
from tqdm import tqdm
import logging
from src.model import Seq2Seq, PointerGeneratedSeq2Seq
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, id, code, code_tokens, criterion, occurrence):
        self.id = id
        self.code = code
        self.code_tokens = code_tokens
        self.criterion = criterion
        self.occurrence = occurrence


def evaluate_single_instance(instance, model, tokenizer, args, use_pointer=False):
    instance = instance.to(args.device)
    with torch.no_grad():
        instance_preds = model(instance.unsqueeze(0))  # Add batch dimension

    text_topk = []
    for topk_pred in instance_preds[0]:  # Only one instance, so take the first one
        t = topk_pred[topk_pred != -999]
        if use_pointer:
            t = instance[t]
        text_topk.append(t)

    if use_pointer:
        item_gold = instance[2]
    else:
        item_gold = instance[1]
    item_gold = item_gold[item_gold != 1]

    item_preds = tokenizer.decode(text_topk[0].tolist(), clean_up_tokenization_spaces=False)
    item_gold = tokenizer.decode(item_gold.tolist()[1: -1], clean_up_tokenization_spaces=False)

    return {'preds_topK': [item_preds], 'gold': item_gold}


def tokenize_and_encode(js: InputExample, tokenizer, args):
    max_source_size, max_target_size = args.max_source_size, args.max_target_size

    # Encoder-Decoder for Trace Generation
    occurrence_tokens = tokenizer.tokenize(str(js.occurrence))
    criterion_token = tokenizer.tokenize(f"<{js.criterion}>")

    source_tokens = js.code_tokens[: max_source_size - 9 - len(occurrence_tokens)]
    source_tokens = ["<s>", "<encoder-decoder>", "</s>"] + source_tokens + ["</s>"] + \
                    criterion_token + ["</s>"] + occurrence_tokens + ["<mask0>", "</s>"]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_padding_length = max_source_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id for _ in range(source_padding_length)]

    # gold_tokens = js.slice_tokens[: max_target_size - 2]
    # gold_tokens = ["<mask0>"] + gold_tokens + ["</s>"]
    # gold_ids = tokenizer.convert_tokens_to_ids(gold_tokens)
    # target_padding_length = max_target_size - len(gold_ids)
    # gold_ids += [tokenizer.pad_token_id for _ in range(target_padding_length)]

    return torch.tensor(source_ids)


def predict(input_text, criterion, occurrence, model, tokenizer, args):
    model.eval()
    # 将输入文本转换为模型可以理解的token
    tokenize = tokenizer.tokenize(input_text)
    js = InputExample(id=0, code=input_text, code_tokens=tokenizer.tokenize(" ".join(tokenize)),
                      criterion=int(criterion), occurrence=occurrence + 1)
    encode = tokenize_and_encode(js, tokenizer, args)
    # print("encode: ", encode)
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = evaluate_single_instance(encode, model, tokenizer, args, use_pointer=False)

    # 将模型的预测结果解码回文本格式
    return outputs


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--encoder", default='unixcoder', type=str,
                        choices=['unixcoder', 'graphcodebert'], help="Encoder in Seq2Seq framework.")
    parser.add_argument("--decoder", default='transformer', type=str,
                        choices=['unixcoder', 'graphcodebert', 'transformer'], help="Decoder in Seq2Seq framework.")

    parser.add_argument("--load_model_path", default='../fine_tuned_model_epoch10/model.ckpt', type=str,
                        help="Path to trained model: Should contain the .bin files")
    # Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--max_source_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_target_size", default=512, type=int,
                        help="Optional output sequence length after tokenization.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="beam size for beam search")
    args = parser.parse_args()
    return args


def setup_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args


def setup_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('../../codeexecutor')
    special_tokens_list = ['<line>', '<state>', '</state>', '<dictsep>', '<output>', '<indent>',
                           '<dedent>', '<mask0>']
    for i in range(200):
        special_tokens_list.append(f"<{i}>")

    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def setup_model(args, tokenizer):
    config = RobertaConfig.from_pretrained('../../codeexecutor')
    encoder = RobertaModel.from_pretrained('../../codeexecutor', config=config)
    encoder.resize_token_embeddings(len(tokenizer))
    decoder_layer = TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads,
        dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob,
        activation=config.hidden_act, layer_norm_eps=config.layer_norm_eps,
    )
    decoder = TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder, encoder_key=args.encoder, decoder=decoder, decoder_key=args.decoder,
        tokenizer=tokenizer, config=config, beam_size=args.beam_size,
        max_source_length=args.max_source_size, max_target_length=args.max_target_size,
        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id,
    )
    model.load_state_dict(torch.load(args.load_model_path), strict=False)
    print("Model loaded")
    model.to(args.device)
    return model


def get_last_processed_line(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            last_line = f.readlines()
            return len(last_line)
    else:
        return 0


def get_delete_line_number(line_id: set, code: str):
    line_number = []
    num = 0
    for code_seg in code.split('\n'):
        for line in line_id:
            if line == code_seg or line.replace('\n', '').strip() in code_seg:
                line_number.append(num)
        num += 1
    return line_number


if __name__ == '__main__':
    args = parse_arguments()
    args = setup_device(args)
    tokenizer = setup_tokenizer()
    model = setup_model(args, tokenizer)
    with open('./vul_function_modified_simplication_need_Dynamic_exec.json', 'r', encoding='utf-8') as f:
        readlines = f.readlines()
        for line in tqdm(readlines[:500:], desc='predict', ncols=100, ascii=True):
            loads = json.loads(line)
            # delete_line = loads['delete_line']
            # pre_function = loads['pre_function']
            pre_function_simplication = loads['pre_function_simplication']
            # 检索delete_line中的行在pre_function_simplication中的位置，以\n为分隔符计数行号，从0开始
            num = 0
            dict_write = {}
            deleted_lines = loads['diff_line_info']['deleted_lines']
            # print("deleted_lines", deleted_lines)
            index = get_delete_line_number(deleted_lines, pre_function_simplication)
            # print("index", index)
            for i in index:
                predict1 = predict(
                    pre_function_simplication,
                    criterion=i,
                    occurrence=1,
                    model=model,
                    tokenizer=tokenizer,
                    args=args
                )
                dict_write[i] = predict1['preds_topK'][0]
                # print(dict_write)
                # torch.cuda.empty_cache()
            loads['DynamicExecutionPath'] = dict_write
            # break
            # print(loads)
            with open('./0_500.json', 'a', encoding='utf-8') as f2:
                f2.writelines(json.dumps(loads, ensure_ascii=False) + '\n')
