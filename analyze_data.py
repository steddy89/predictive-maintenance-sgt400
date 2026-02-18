import openpyxl
import json
from collections import Counter

path = r'C:\Users\teddysudewo\OneDrive - Microsoft\TeddyS\My_git_project\applying-asynchronous-programming-c-sharp\temp_data.xlsx'
wb = openpyxl.load_workbook(path, read_only=True)
ws = wb['Sheet1']

rows = list(ws.iter_rows(min_row=2, values_only=True))  # skip header
print(f"Total respondents: {len(rows)}")

# Parse all unique programs
all_programs = Counter()
respondents = []
for row in rows:
    rid, name, programs_str = row
    if programs_str:
        progs = [p.strip() for p in programs_str.split(';') if p.strip()]
        for p in progs:
            all_programs[p] += 1
        respondents.append({'id': rid, 'name': name, 'programs': progs})

print(f"\nUnique programs ({len(all_programs)}):")
for prog, count in all_programs.most_common():
    print(f"  {count:4d} | {prog}")

print(f"\nSample entries:")
for r in respondents[:5]:
    print(f"  {r['id']}: {r['name']} -> {r['programs']}")

# Check how many programs each person selected
from collections import Counter as C2
selection_counts = C2(len(r['programs']) for r in respondents)
print(f"\nSelections per person: {dict(sorted(selection_counts.items()))}")

wb.close()
