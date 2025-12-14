"""Making sure ChatGPT did not fuck up when generating answer options :)"""

import re
import json
from pathlib import Path

with open("questions.json") as in_fie:
    generated_questions = {re.sub(r"[^\w\s]", "", q["question"].lower().strip()) for q in json.load(in_fie)}

real_questions = {re.sub(r"[^\w\s]", "", m.lower().strip()): m for m in
                  re.findall(r"(?<=\*\*Вопрос:\*\* ).*", Path("Final_test_questions.md").read_text())}

extra = generated_questions - set(real_questions)
print("Hallucinated:", len(extra), sorted(extra))
missing = set(real_questions) - generated_questions
print("Missing:", len(missing), sorted(real_questions[m] for m in missing))

print(f"{len(real_questions)=}")
print(f"{len(generated_questions)=}")
