import json

key_sums = {}
with open("/home/zhangyusi/raptor/output_result/qasper-test-v0.3_llama3.1_nomic-embed-text/raptor_llm-familiar-filter-6-3_cnt_reduce.json", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        for key, value in data.items():
            if isinstance(value, int):
                if key in key_sums:
                    key_sums[key] += value
                else:
                    key_sums[key] = value

for key, total in key_sums.items():
    print(f"{key}: {total}")

print('\n')
print(f"Removed {key_sums["remove_text_len"]}/{key_sums["total_text_len"]} text length, reduced {key_sums["remove_text_len"]/key_sums["total_text_len"] * 100:.2f}%.")
print(f"Removed {key_sums["remove_text_token"]}/{key_sums["total_text_token"]} text token, reduced {key_sums["remove_text_token"]/key_sums["total_text_token"] * 100:.2f}%.")
# print(f"Removed {key_sums["remove_hypo_qs_len"]}/{key_sums["total_hypo_qs_len"]} hypo question length, reduced {key_sums["remove_hypo_qs_len"]/key_sums["total_hypo_qs_len"] * 100:.2f}%.")
# print(f"Removed {key_sums["remove_hypo_qs_token"]}/{key_sums["total_hypo_qs_token"]} hypo question token, reduced {key_sums["remove_hypo_qs_token"]/key_sums["total_hypo_qs_token"] * 100:.2f}%.")
print(f"Removed {key_sums["remove_node"]}/{key_sums["total_node"]} nodes, reduced {key_sums["remove_node"]/key_sums["total_node"] * 100:.2f}%.")
