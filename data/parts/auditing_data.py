import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Open the file and load the data
with open("se_pairs_v2.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# print(data[1])
num_of_rows_to_check = 50
random_rows = random.sample(data, k=num_of_rows_to_check)
# print(random_rows[1])

with open("output_v2.txt", "w", encoding="utf-8") as f:
    for row in random_rows:
        f.write("\n")
        f.write("=" * 120)
        f.write(f"\nprompt:\n{row['prompt']}")
        f.write(f"\nchoosen:\n{row['chosen']}")
        f.write(f"\nrejected:\n{row['rejected']}")
