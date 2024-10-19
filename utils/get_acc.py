import json

total = 0
correct = 0
with open("/home/zhangyusi/raptor/output_result/dev_llama3.1_nomic-embed-text/raptor_cosine-sim-filter-0-80.json", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        if data["is_correct"]:
            correct += 1
        total += 1
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Acc: {correct/total * 100:.2f}%")