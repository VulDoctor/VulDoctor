import json
import os

from tqdm import tqdm


def location(lines: list, pre_code: str):
    return_lines = []
    line_number = 1
    for line in lines:
        for code_seg in pre_code.split('\n'):
            if line == code_seg or line.replace('\n', '') in code_seg:
                return_lines.append(line_number)
            line_number += 1
    return return_lines


def get_line_id(line_numbers: list, nodes: list):
    line_id = []
    for line in line_numbers:
        for node in nodes:
            if 'lineNumber' in node.keys() and line == node['lineNumber']:
                line_id.append(node['id'])
    return line_id


def get_forward_slice(line_id: list, edges: list, slice_depth: int = 5):
    forward_slice = set()
    for line in line_id:
        line_forward = set()
        for i in range(slice_depth):
            for edge in edges:
                if edge['inNode'] == line:
                    line_forward.add(edge['outNode'])
        forward_slice.update(line_forward)
    return forward_slice


def get_backward_slice(line_id: list, edges: list, slice_depth: int = 5):
    backward_slice = set()
    for line in line_id:
        line_forward = set()
        for i in range(slice_depth):
            for edge in edges:
                if edge['outNode'] == line:
                    line_forward.add(edge['inNode'])
        backward_slice.update(line_forward)
    return backward_slice


def get_id_line_number(line_id: set, nodes: list):
    line_number = []
    for line in line_id:
        for node in nodes:
            if 'id' in node.keys() and line == node['id'] and 'lineNumber' in node.keys():
                line_number.append(node['lineNumber'])
    return line_number


def get_delete_slice(line_id: set, code: str):
    slice_lines = []
    for line in line_id:
        slice_lines.append(code.split('\n')[line - 1])
    return slice_lines


def get_index_number(index_list: list, code: str):
    num = 0
    num_list = []
    for code_seg in code.split('\n'):
        if code_seg in index_list:
            num_list.append(num)
        num += 1
    return num_list


if __name__ == '__main__':
    # dict_1 =     {'cve_id': 'CVE-2012-1179', 'cwe_id': ['CWE-264'], 'cvss_vector': 'AV:A/AC:M/Au:S/C:N/I:N/A:C', 'repo_name': 'torvalds/linux', 'func': 'static void mincore_pmd_range(struct vm_area_struct *vma, pud_t *pud,\n\t\t\tunsigned long addr, unsigned long end,\n\t\t\tunsigned char *vec)\n{\n\tunsigned long next;\n\tpmd_t *pmd;\n\n\tpmd = pmd_offset(pud, addr);\n\tdo {\n\t\tnext = pmd_addr_end(addr, end);\n\t\tif (pmd_trans_huge(*pmd)) {\n\t\t\tif (mincore_huge_pmd(vma, pmd, addr, next, vec)) {\n\t\t\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t\t\t\tcontinue;\n\t\t\t}\n\t\t\t/* fall through */\n\t\t}\n\t\tif (pmd_none_or_trans_huge_or_clear_bad(pmd))\n\t\t\tmincore_unmapped_range(vma, addr, next, vec);\n\t\telse\n\t\t\tmincore_pte_range(vma, pmd, addr, next, vec);\n\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t} while (pmd++, addr = next, addr != end);\n}', 'func_before': 'static void mincore_pmd_range(struct vm_area_struct *vma, pud_t *pud,\n\t\t\tunsigned long addr, unsigned long end,\n\t\t\tunsigned char *vec)\n{\n\tunsigned long next;\n\tpmd_t *pmd;\n\n\tpmd = pmd_offset(pud, addr);\n\tdo {\n\t\tnext = pmd_addr_end(addr, end);\n\t\tif (pmd_trans_huge(*pmd)) {\n\t\t\tif (mincore_huge_pmd(vma, pmd, addr, next, vec)) {\n\t\t\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t\t\t\tcontinue;\n\t\t\t}\n\t\t\t/* fall through */\n\t\t}\n\t\tif (pmd_none_or_clear_bad(pmd))\n\t\t\tmincore_unmapped_range(vma, addr, next, vec);\n\t\telse\n\t\t\tmincore_pte_range(vma, pmd, addr, next, vec);\n\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t} while (pmd++, addr = next, addr != end);\n}', 'diff_func': '--- func_before\n+++ func_after\n@@ -15,7 +15,7 @@\n \t\t\t}\n \t\t\t/* fall through */\n \t\t}\n-\t\tif (pmd_none_or_clear_bad(pmd))\n+\t\tif (pmd_none_or_trans_huge_or_clear_bad(pmd))\n \t\t\tmincore_unmapped_range(vma, addr, next, vec);\n \t\telse\n \t\t\tmincore_pte_range(vma, pmd, addr, next, vec);', 'diff_line_info': {'deleted_lines': ['\t\tif (pmd_none_or_clear_bad(pmd))'], 'added_lines': ['\t\tif (pmd_none_or_trans_huge_or_clear_bad(pmd))']}, 'is_vul': True, 'commit_hash': '4a1d704194a441bf83c636004a479e01360ec850', 'parent_commit_hash': 'a998dc2fa76f496d2944f0602b920d1d10d7467d', 'func_graph_path_before': 'torvalds/linux/4a1d704194a441bf83c636004a479e01360ec850/mincore.c/vul/before/0.json', 'delete_slice': [], 'pre_function_simplication': 'static void mincore_pmd_range(struct vm_area_struct *vma, pud_t *pud,\n\t\t\tunsigned long addr, unsigned long end,\n\t\t\tunsigned char *vec)\n{\n}', 'post_function_simplication': 'static void mincore_pmd_range(struct vm_area_struct *vma, pud_t *pud,\n\t\t\tunsigned long addr, unsigned long end,\n\t\t\tunsigned char *vec)\n{\n\tdo {\n\t\tnext = pmd_addr_end(addr, end);\n\t\tif (pmd_trans_huge(*pmd)) {\n\t\t\tif (mincore_huge_pmd(vma, pmd, addr, next, vec)) {\n\t\t\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t\t\t\tcontinue;\n\t\t\t}\n\t\t\t/* fall through */\n\t\t}\n\t\tif (pmd_none_or_trans_huge_or_clear_bad(pmd))\n\t\t\tmincore_unmapped_range(vma, addr, next, vec);\n\t\telse\n\t\t\tmincore_pte_range(vma, pmd, addr, next, vec);\n\t\tvec += (next - addr) >> PAGE_SHIFT;\n\t} while (pmd++, addr = next, addr != end);\n}'}
    # print(os.path.join('./megavul_graph', dict_1['func_graph_path_before']))
    # with open(os.path.join('./megavul_graph', dict_1['func_graph_path_before']), 'r', encoding='utf-8') as f:
    #     graph = json.loads(f.read())
    #     print('line_number', get_index_number(dict_1['diff_line_info']['deleted_lines'], dict_1['func_before']))
    #     line_id = get_line_id(
    #         get_index_number(dict_1['diff_line_info']['deleted_lines'], dict_1['func_before']),
    #         graph['nodes']
    #     )
    #     forward_result = get_forward_slice(line_id, graph['edges'])
    #     backward_result = get_backward_slice(line_id, graph['edges'])
    #     print(forward_result, backward_result)
    #     # 将line_id与forward_result和backward_result合并去重
    #     line_id = set(line_id)
    #     line_id.update(forward_result)
    #     line_id.update(backward_result)
    #     number = get_id_line_number(line_id, graph['nodes'])
    #     delete_slice = get_delete_slice(set(number), dict_1['func_before'])
    #     print(dict_1['diff_line_info']['deleted_lines'])
    #     print("delete_slice", delete_slice)
    # dict_1['delete_slice'] = delete_slice
    depth = [1, 3, 5, 10]
    num = 0
    error_number = 0
    for dep in depth:
        with open('./wait_for_simplication_only_delete_and_add.json', 'r', encoding='utf-8') as f:
            readlines = f.readlines()
            for line in tqdm(readlines, desc='simplication', ncols=100, ascii=True):
                loads = json.loads(line)
                try:
                    with open(os.path.join('./megavul_graph', loads['func_graph_path_before']), 'r',
                              encoding='utf-8') as f1:
                        graph = json.loads(f1.read())
                        line_id = get_line_id(
                            get_index_number(loads['diff_line_info']['deleted_lines'], loads['func_before']),
                            graph['nodes']
                        )
                        forward_result = get_forward_slice(line_id, graph['edges'], slice_depth=dep)
                        backward_result = get_backward_slice(line_id, graph['edges'], slice_depth=dep)
                        # 将line_id与forward_result和backward_result合并去重
                        line_id = set(line_id)
                        line_id.update(forward_result)
                        line_id.update(backward_result)
                        number = get_id_line_number(line_id, graph['nodes'])
                        delete_slice = get_delete_slice(set(number), loads['func_before'])
                        # 将delete_slice与loads['diff_line_info']['deleted_lines']两个列表合并并去重
                        delete_slice.extend(loads['diff_line_info']['deleted_lines'])
                        delete_slice = list(set(delete_slice))
                        loads['delete_slice'] = list(delete_slice)
                except Exception as e:
                    error_number += 1
                    continue
                if not os.path.exists('./depth_'+str(dep)):
                    os.mkdir('./depth_'+str(dep))
                with open('./depth_'+str(dep)+'/wait_for_simplication_only_delete_and_add_with_slice.json', 'a', encoding='utf-8') as f2:
                    f2.write(json.dumps(loads) + '\n')

        print(error_number)
        print(num)
