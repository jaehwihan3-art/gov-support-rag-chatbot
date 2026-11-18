"""
data/answer_key/answer_key_NN_draft.json -> answer_key_NN.json 로 이름을 바꾸고 _상태를 "확정"으로 바꾼다.
이미 확정본(answer_key_NN.json)이 있으면 그 번호는 건드리지 않는다.

사용법 (govchat 폴더에서):
    python finalize_keys.py
"""
import glob, json, os, re

d = os.path.join("data", "answer_key")
done = skipped = 0
for path in sorted(glob.glob(os.path.join(d, "answer_key_*_draft.json"))):
    m = re.search(r"answer_key_(\d\d)_draft\.json$", path)
    if not m:
        continue
    new = os.path.join(d, f"answer_key_{m.group(1)}.json")
    if os.path.exists(new):
        print(f"건너뜀: {os.path.basename(new)} 이미 있음")
        skipped += 1
        continue
    data = json.load(open(path, encoding="utf-8"))
    data["_상태"] = "확정"
    with open(new, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.remove(path)
    done += 1
print(f"확정 처리 {done}개, 건너뜀 {skipped}개")
