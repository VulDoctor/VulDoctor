import json
import difflib
import random

from tqdm import tqdm


def diff(pre, post):
    pre = pre.split('\n')
    post = post.split('\n')
    diff = difflib.ndiff(pre, post)
    return diff


# def get_pre_and_post_data(diff):
#     pre = []
#     post = []
#     current_pre = []
#     current_post = []
#     for line in diff:
#         if line.startswith('-'):
#             if current_post:  # If we were processing added lines, wrap them up
#                 post.append('<vul-start>\n' + '\n'.join(current_post) + '\n<vul-end>')
#                 current_post = []
#             current_pre.append(line[2:])  # Remove the '- ' prefix
#         elif line.startswith('+'):
#             if current_pre:  # If we were processing deleted lines, wrap them up
#                 pre.append('<vul-start>\n' + '\n'.join(current_pre) + '\n<vul-end>')
#                 current_pre = []
#             current_post.append(line[2:])  # Remove the '+ ' prefix
#         elif line.startswith(' '):  # If the line has not changed, add it to pre
#             if current_pre:  # If we were processing deleted lines, wrap them up
#                 pre.append('<vul-start>\n' + '\n'.join(current_pre) + '\n<vul-end>')
#                 current_pre = []
#             pre.append(line[2:])  # Remove the '  ' prefix
#     # Wrap up any remaining lines
#     if current_pre:
#         pre.append('<vul-start>\n' + '\n'.join(current_pre) + '\n<vul-end>')
#     if current_post:
#         post.append('<vul-start>\n' + '\n'.join(current_post) + '\n<vul-end>')
#     return pre, post
def get_pre_and_post_data(diff_lines):
    pre = []
    post = []

    for line in diff_lines:
        if line.startswith('?'):
            continue
        if line.startswith('  ') or line.replace('\n', '').replace('\t', '').strip() == '':
            pre.append(line[2:])
            post.append(line[2:])
        elif line.startswith('+ '):
            post.append('<vul-start> ' + line[2:] + '<vul-end>')
        elif line.startswith('- '):
            pre.append('<vul-start> ' + line[2:] + '<vul-end>')

    return '\n'.join(pre), '\n'.join(post)

def write_json(readfilepath, writefilepath, mode):
    all_dataset = []
    with open(readfilepath, 'r', encoding='utf-8') as f:
        readlines = f.readlines()
        for line in tqdm(readlines, desc=mode, total=len(readlines)):
            loads = json.loads(line)
            # if '{}' in loads['pre_function_simplication'].replace('\n', '').replace('\t', '').replace(' ', ''):
            #     pre_post_diff = diff(loads['pre_function'], loads['post_function'])
            # else:
            pre, post = get_pre_and_post_data(diff(loads['pre_function_simplication'], loads['post_function_simplication']))
            if ''.join(post).replace('\n', '').replace('<vul-start>', '').replace('<vul-end>', '').strip() == '':
                continue
            post = post[post.find('<vul-start>'):post.rfind('<vul-end>') + len('<vul-end>')]
            loads['question'] = replace_space(pre)
            loads['answer'] = replace_space(post)
            all_dataset.append(loads)
            # all_dataset.append({
            #     'question': ''.join(pre).replace('\n', '').replace('\t', ''),
            #     'answer': ''.join(post).replace('\n', '').replace('\t', ''),
            #     'dynamic_execution_path': loads['DynamicExecutionPath']
            # })
    with open(writefilepath, 'a', encoding='utf-8') as f1:
        f1.write(json.dumps(all_dataset, ensure_ascii=False, indent=4))


# 将字符串中连续的若干个空格替换为一个空格
def replace_space(s):
    s = s.replace('\t', ' ')
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s


if __name__ == '__main__':
    with open('./total.json', 'r', encoding='utf-8') as f:
        readlines = f.readlines()
        random.shuffle(readlines)
        train = readlines[:int(len(readlines) * 0.7)]
        dev = readlines[int(len(readlines) * 0.7):int(len(readlines) * 0.8)]
        test = readlines[int(len(readlines) * 0.8):]
        with open('./train.json', 'a', encoding='utf-8') as f1:
            for line in tqdm(train, desc='train', ncols=100, ascii=True):
                loads = json.loads(line)
                f1.writelines(json.dumps(loads, ensure_ascii=False) + '\n')
        with open('./dev.json', 'a', encoding='utf-8') as f2:
            for line in tqdm(dev, desc='dev', ncols=100, ascii=True):
                json_loads = json.loads(line)
                f2.writelines(json.dumps(json_loads, ensure_ascii=False) + '\n')
        with open('./test.json', 'a', encoding='utf-8') as f3:
            for line in tqdm(test, desc='test', ncols=100, ascii=True):
                loads1 = json.loads(line)
                f3.writelines(json.dumps(loads1, ensure_ascii=False) + '\n')
        print(len(train), len(dev), len(test))
    write_json('./train.json', '../train_dataset.json', 'train')
    write_json('./dev.json', '../dev_dataset.json', 'dev')
    write_json('./test.json', '../test_dataset.json', 'test')

    # code = '''<vul-start>                                   PREDICTION_MODE mode, TX_SIZE tx_size,<vul-end><vul-start>DECLARE_ALIGNED(16, uint8_t, left_col[32]);<vul-end><vul-start>DECLARE_ALIGNED(16, uint8_t, above_data[64 + 16]);<vul-end><vul-start>if (extend_modes[mode] & NEED_LEFT) {<vul-end><vul-start>    if (left_available) {<vul-end><vul-start>      if (xd->mb_to_bottom_edge < 0) {<vul-end><vul-start>        /* slower path if the block needs border extension */<vul-end><vul-start>        if (y0 + bs <= frame_height) {          for (i = 0; i < bs; ++i)            left_col[i] = ref[i * ref_stride - 1];        } else {          const int extend_bottom = frame_height - y0;          for (i = 0; i < extend_bottom; ++i)            left_col[i] = ref[i * ref_stride - 1];          for (; i < bs; ++i)            left_col[i] = ref[(extend_bottom - 1) * ref_stride - 1];        }      } else {        /* faster path if the block does not need extension */<vul-end><vul-start>      memset(left_col, 129, bs);<vul-end><vul-start>if (extend_modes[mode] & NEED_ABOVE) {<vul-end><vul-start>    if (up_available) {<vul-end><vul-start>      const uint8_t *above_ref = ref - ref_stride;<vul-end><vul-start>      if (xd->mb_to_right_edge < 0) {<vul-end><vul-start>        /* slower path if the block needs border extension */<vul-end><vul-start>        if (x0 + bs <= frame_width) {<vul-end><vul-start>          memcpy(above_row, above_ref, bs);        } else if (x0 <= frame_width) {          const int r = frame_width - x0;          memcpy(above_row, above_ref, r);          memset(above_row + r, above_row[r - 1], x0 + bs - frame_width);        }      } else {        /* faster path if the block does not need extension */        if (bs == 4 && right_available && left_available) {          const_above_row = above_ref;<vul-end><vul-start>          memcpy(above_row, above_ref, bs);<vul-end><vul-start>      memset(above_row, 127, bs);      above_row[-1] = 127;    }  }if (extend_modes[mode] & NEED_ABOVERIGHT) {    if (up_available) {      const uint8_t *above_ref = ref - ref_stride;      if (xd->mb_to_right_edge < 0) {<vul-end><vul-start>        /* slower path if the block needs border extension */<vul-end><vul-start>        if (x0 + 2 * bs <= frame_width) {          if (right_available && bs == 4) {            memcpy(above_row, above_ref, 2 * bs);          } else {            memcpy(above_row, above_ref, bs);            memset(above_row + bs, above_row[bs - 1], bs);          }        } else if (x0 + bs <= frame_width) {          const int r = frame_width - x0;          if (right_available && bs == 4) {            memcpy(above_row, above_ref, r);            memset(above_row + r, above_row[r - 1], x0 + 2 * bs - frame_width);          } else {            memcpy(above_row, above_ref, bs);            memset(above_row + bs, above_row[bs - 1], bs);          }        } else if (x0 <= frame_width) {          const int r = frame_width - x0;          memcpy(above_row, above_ref, r);          memset(above_row + r, above_row[r - 1], x0 + 2 * bs - frame_width);        }        /* faster path if the block does not need extension */        if (bs == 4 && right_available && left_available) {          const_above_row = above_ref;        } else {<vul-end><vul-start>          memcpy(above_row, above_ref, bs);<vul-end><vul-start>          if (bs == 4 && right_available)<vul-end><vul-start>            memcpy(above_row + bs, above_ref + bs, bs);<vul-end><vul-start>          else<vul-end><vul-start>            memset(above_row + bs, above_row[bs - 1], bs);<vul-end><vul-start>        }      above_row[-1] = left_available ? above_ref[-1] : 129;    } else {      memset(above_row, 127, bs * 2);      above_row[-1] = 127;<vul-end>'''
    # space = replace_space(code)
    # print(space)
