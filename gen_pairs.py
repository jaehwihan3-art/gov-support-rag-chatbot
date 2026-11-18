"""
확정 정답표(data/answer_key/answer_key_NN.json) x 가상 프로필 -> '프로필 x 공고' 쌍 300개와 정답 판정을 코드로 만든다.

원칙
- 정답 판정(gold)은 사람이 원문으로 확정한 정답표 + gen_rules.json만 보고 계산한다 (LLM 결과는 안 씀).
- 공고마다 10쌍. 불가 약 50%, 조건부는 전체 10% 이하, 나머지 가능.
- 불가 프로필은 '조건 하나만' 어긴다 -> 오답이 어느 조건에서 나는지 추적 가능.
- 경계값(경계 안/밖)을 일부러 넣는다: 나이, 출생연도, 업력, 개업일, 매출, 신용점수.
- 프로필에는 모든 기타 필수 자격의 답과 '그 외 제외 사유 없음'을 적어서, 정보 부족이 아닌 쌍은 정답이 확정되게 한다.
- 홀드아웃은 공고 단위로 나눈다. 1~3번(seen_before)은 개발셋에만 둔다.
- 평가 전에 정답 분포를 먼저 출력한다.

사용법 (govchat 폴더에서):
    python gen_pairs.py
    python gen_pairs.py --keys data/answer_key --rules gen_rules.json --out data/pairs --seed 42
출력: data/pairs/pairs_all.csv, pairs_dev.csv, pairs_holdout.csv, gold_report.md
"""
import argparse, csv, glob, json, os, random, re
from datetime import date, timedelta

POSSIBLE, IMPOSSIBLE, CONDITIONAL = "신청 가능", "신청 불가", "조건부 가능"
SIZES = ["예비창업자", "소상공인", "소기업(소상공인 아님)", "중기업", "중견기업"]
SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기",
        "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
GU = {"대전": ["동구", "중구", "서구", "유성구", "대덕구"], "서울": ["강남구", "마포구", "강서구", "관악구"],
      "부산": ["해운대구", "부산진구"], "대구": ["수성구", "달서구"], "인천": ["연수구", "남동구"],
      "광주": ["북구", "광산구"], "울산": ["남구"], "경기": ["성남시", "수원시", "화성시"],
      "세종": ["세종시"], "강원": ["춘천시"], "충북": ["청주시"], "충남": ["천안시"], "전북": ["전주시"],
      "전남": ["순천시"], "경북": ["포항시"], "경남": ["창원시"], "제주": ["제주시"]}

# 업종 풀 (KSIC 대분류 문자 + 숫자)
TECH = [("C26299", "기타 전자부품 제조업"), ("J58222", "응용 소프트웨어 개발 및 공급업"),
        ("M70113", "전기·전자공학 연구개발업"), ("J62010", "컴퓨터 프로그래밍 서비스업"),
        ("J63120", "포털 및 기타 인터넷 정보매개 서비스업"), ("C29199", "그 외 기타 일반 목적용 기계 제조업")]
GENERAL = [("I56111", "한식 일반 음식점업"), ("G47190", "그 외 기타 종합 소매업"), ("S96112", "두발 미용업"),
           ("S95120", "통신장비 수리업"), ("C10712", "떡류 제조업"), ("C25929", "그 외 기타 금속 가공제품 제조업")]
EXCLUDED = [("I56211", "일반유흥 주점업"), ("I56212", "무도유흥 주점업"), ("R91249", "기타 사행시설 관리 및 운영업")]
OUTSIDE = [("I56111", "한식 일반 음식점업"), ("G47190", "그 외 기타 종합 소매업"), ("S96112", "두발 미용업")]
TECH_SECTIONS = set("CJMNPQ")       # 14번 트랙2 기술기반업종 (제조·정보통신·전문과학기술·사업지원·교육·보건)
# 업종 성격의 제외 사유: 프로필 업종도 그 사유에 맞게 바꾼다 (예: '단순 카페' 해당인데 업종이 소프트웨어면 모순)
# 업종 제외 표 공통 코드(EXCLUDED)는 피해서 '조건 하나만 어긴' 쌍이 유지되게 함
FLAG_KSIC = {
    "단순_카페_도소매_유통": {"ksic": "I56221", "ksic_name": "커피 전문점"},
    "무점포_비접객_업종": {"ksic": "G47911", "ksic_name": "전자상거래 소매업"},
    "사치향락_투기조장_업종": {"ksic": "R91291", "ksic_name": "무도장 운영업"},
}
INDUSTRY_FLAGS = {"제외업종", "창업제외업종", "융자제외업종"}   # 업종 코드로 판정하는 제외 사유
SIZE_FLAGS = {"소상공인"}                                    # 16번: 기업규모로 판정
REV_RANGE = {"소상공인": (30_000_000, 900_000_000), "소기업(소상공인 아님)": (1_000_000_000, 8_000_000_000),
             "중기업": (12_000_000_000, 60_000_000_000), "중견기업": (150_000_000_000, 400_000_000_000)}


REF = [date(2026, 11, 11)]   # 판정 기준일 (main에서 gen_rules.json 값으로 바꿈)


# ---------------- 날짜 도우미 ----------------
def months_between(d0, d1):
    m = (d1.year - d0.year) * 12 + d1.month - d0.month
    return m - (1 if d1.day < d0.day else 0)


def minus_months(d, m):
    y, mo = divmod(d.year * 12 + d.month - 1 - m, 12)
    mo += 1
    for day in (d.day, 28):
        try:
            return date(y, mo, day)
        except ValueError:
            continue


def age_on(birth, ref):
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


# ---------------- 정답 판정 ----------------
class Judge:
    def __init__(self, n, rule, ref):
        self.n, self.r, self.ref = n, rule, ref

    def allowed_types(self, p):
        t = set(self.r["types"])
        if p.get("ksic") and p["ksic"].startswith("C"):
            t |= set(self.r.get("types_if_manufacturing", []))
        return t

    def computed(self, key, p):
        """프로필 값으로 답이 정해지는 기타 자격. None = 정보 부족."""
        cr = self.r.get("credit", {})
        nice, kcb = p.get("nice"), p.get("kcb")
        if key == "사업자등록_보유":
            return p["size"] != "예비창업자"
        if key == "신용점수":
            if nice is None and kcb is None:
                return None
            return (nice is not None and nice >= cr["nice_min"]) or (kcb is not None and kcb >= cr["kcb_min"])
        if key == "저신용_또는_저소득":
            low = (nice is not None and nice <= cr["nice_max"]) or (kcb is not None and kcb <= cr["kcb_max"])
            if low or p.get("저소득"):
                return True
            if nice is None or kcb is None or p.get("저소득") is None:
                return None
            return False
        if key == "개인사업자":
            return p.get("biz_type") == "개인"
        if key == "여성기업":
            return None if p.get("gender") is None else p["gender"] == "여"
        if key == "트랙_해당":       # 14번: ①유성구 3개월 이상 영업 또는 ②개업 3년 미만 + 기술기반업종
            m = p.get("biz_months")
            if m is None:
                return None
            return m >= 3 or (m < 36 and p["ksic"][0] in TECH_SECTIONS)
        raise KeyError(key)

    def __call__(self, p):
        n, fails, unknown = self.n, [], []
        pre = p["size"] == "예비창업자"

        # 1) 신청 가능한 기업 규모 (법률 용어: 예비창업자·소상공인·중소기업 등)
        if p["size"] not in self.allowed_types(p):
            fails.append("기업규모")

        # 2) 대표자 나이·출생연도
        a = n.get("age")
        if a:
            if p.get("birth") is None:
                unknown.append("age")
            else:
                age = age_on(p["birth"], self.ref)
                if (a.get("min") is not None and age < a["min"]) or (a.get("max") is not None and age > a["max"]):
                    fails.append("age")
                if a.get("birth_year_min") is not None and p["birth"].year < a["birth_year_min"]:
                    fails.append("birth_year")

        # 3) 지역 ('대전' 또는 '대전 중구' 형식)
        reg = n.get("region") or {}
        if reg.get("allowed"):
            if p.get("sido") is None:
                unknown.append("region")
            else:
                ok = any(p["sido"] == a_.split()[0] and (len(a_.split()) == 1 or p.get("gu") == a_.split()[1])
                         for a_ in reg["allowed"])
                if not ok:
                    fails.append("region")

        # 4) 업종
        k = p["ksic"]
        if self.r.get("allowed_ksic") and not any(k.startswith(x) for x in self.r["allowed_ksic"]):
            fails.append("industry")
        if any(k.startswith(x) for x in self.r.get("excluded_ksic", [])):
            fails.append("industry")
        if self.r.get("excl_table") and k in {c for c, _ in EXCLUDED}:
            fails.append("industry")

        # 5) 업력·개업일 (예비창업자는 기업규모에서 이미 판정)
        b = n.get("biz_months")
        if b and not pre:
            m, opened = p.get("biz_months"), p.get("opened")
            if m is None:
                unknown.append("biz_months")
            else:
                if b.get("min") is not None and m < b["min"]:
                    fails.append("biz_months")
                if b.get("max") is not None and m > b["max"]:
                    exc = None
                    if b.get("max_신산업") is not None and m <= b["max_신산업"]:
                        exc = ("신산업_분야", "biz_months(신산업)")      # 84~119개월: 신산업이면 가능
                    elif b.get("max_면제"):
                        exc = ("업력면제_대상", "biz_months(면제)")       # 업력 상한 면제 대상이면 가능
                    if exc:
                        ans = p["answers"].get(exc[0])
                        if ans is None:
                            unknown.append(exc[1])
                        elif not ans:
                            fails.append(exc[1])
                    else:
                        fails.append("biz_months")
                if b.get("founded_after") and opened < date.fromisoformat(b["founded_after"]):
                    fails.append("founded_after")
                if b.get("founded_before") and opened > date.fromisoformat(b["founded_before"]):
                    fails.append("founded_before")

        # 6) 매출 (전년도 연매출)
        rv = n.get("revenue") or {}
        if (rv.get("min") is not None or rv.get("max") is not None) and not pre:
            v = p.get("revenue")
            if v is None:
                unknown.append("revenue")
            elif (rv.get("min") is not None and v < rv["min"]) or (rv.get("max") is not None and v > rv["max"]):
                fails.append("revenue")

        # 7) 신청 제외 사유
        for ex in n.get("exclusions") or []:
            f = ex["flag"]
            if f in INDUSTRY_FLAGS or f in SIZE_FLAGS:
                continue
            if f in p["flags"]:
                (unknown if "조건부" in ex.get("_메모", "") else fails).append("제외:" + f)

        # 8) 기타 필수 자격
        comp = set(self.r.get("computed", []))
        for req in n.get("other_requirements") or []:
            key = req["key"]
            ans = self.computed(key, p) if key in comp else p["answers"].get(key)
            if ans is None:
                unknown.append("기타:" + key)
            elif not ans:
                fails.append("기타:" + key)

        if fails:
            return IMPOSSIBLE, fails
        if unknown:
            return CONDITIONAL, unknown
        return POSSIBLE, []


# ---------------- 프로필 생성 ----------------
class Maker:
    def __init__(self, n, rule, ref, rng):
        self.n, self.r, self.ref, self.rng = n, rule, ref, rng
        self.judge = Judge(n, rule, ref)

    def pick_ksic(self):
        al = self.r.get("allowed_ksic")
        pool = TECH + GENERAL
        if al:
            cands = [x for x in pool if any(x[0].startswith(a) for a in al)]
        else:
            cands = TECH if self.r.get("tech") else GENERAL
        return self.rng.choice(cands)

    def base(self, size=None):
        rng, n, r = self.rng, self.n, self.r
        p = {"flags": [], "answers": {}, "저소득": False}
        p["ksic"], p["ksic_name"] = self.pick_ksic()
        types = [t for t in r["types"]] + (r.get("types_if_manufacturing", []) if p["ksic"].startswith("C") else [])
        rv = n.get("revenue") or {}
        lo_r, hi_r = rv.get("min"), rv.get("max")
        ok_sizes = [s for s in types if s == "예비창업자" or
                    (hi_r is None or REV_RANGE[s][0] <= hi_r)]
        p["size"] = size or rng.choice(ok_sizes)

        # 대표자
        a = n.get("age") or {}
        amin, amax = a.get("min") or 25, a.get("max") or 62
        if a.get("birth_year_min"):
            amax = min(amax, self.ref.year - a["birth_year_min"] - 1)
        age = rng.randint(amin + 1, max(amin + 1, amax - 1))
        p["birth"] = self.birth_for_age(age)
        p["gender"] = rng.choice(["남", "여"])
        if "여성기업" in r.get("computed", []):
            p["gender"] = "여"

        # 지역
        allowed = (n.get("region") or {}).get("allowed")
        if allowed:
            a_ = rng.choice(allowed).split()
            p["sido"] = a_[0]
            p["gu"] = a_[1] if len(a_) > 1 else rng.choice(GU[a_[0]])
        else:
            p["sido"] = rng.choice(SIDO)
            p["gu"] = rng.choice(GU[p["sido"]])

        # 업력·매출
        if p["size"] == "예비창업자":
            p["opened"] = p["biz_months"] = p["revenue"] = None
            p["biz_type"] = None
        else:
            b = n.get("biz_months") or {}
            lo = b.get("min") or 0
            hi = b.get("max") if b.get("max") is not None else 150
            if b.get("founded_after"):
                hi = min(hi, months_between(date.fromisoformat(b["founded_after"]), self.ref))
            if b.get("founded_before"):
                lo = max(lo, months_between(date.fromisoformat(b["founded_before"]), self.ref) + 1)
            if (lo_r is not None or hi_r is not None) and not b.get("founded_after"):
                # 매출 조건이 있는 공고는 2025년 매출이 있도록 2025년 이전 개업으로 (2026년 개업은 매출 0원)
                lo = max(lo, months_between(date(2025, 12, 31), self.ref) + 1)
            self.set_months(p, rng.randint(lo + 1, max(lo + 1, hi - 1)) if hi - lo > 2 else lo + (hi > lo))
            r0, r1 = REV_RANGE[p["size"]]
            if lo_r is not None:
                r0 = max(r0, lo_r)
            if hi_r is not None:
                r1 = min(r1, hi_r)
            p["revenue"] = round(rng.uniform(r0, max(r0, r1 * 0.9)), -6) or r0
            p["biz_type"] = "개인" if p["size"] == "소상공인" and rng.random() < 0.7 else "법인"
            if "개인사업자" in r.get("computed", []):
                p["biz_type"] = "개인"

        # 신용점수
        cr = r.get("credit", {})
        if "nice_max" in cr:
            p["nice"], p["kcb"] = rng.randint(600, cr["nice_max"] - 20), rng.randint(560, cr["kcb_max"] - 20)
        else:
            p["nice"], p["kcb"] = rng.randint(720, 950), rng.randint(680, 950)

        # 기타 필수 자격: 직접 답하는 문항은 모두 '예'
        comp = set(r.get("computed", []))
        for req in n.get("other_requirements") or []:
            if req["key"] not in comp:
                p["answers"][req["key"]] = True
        return self.normalize(p)

    def birth_for_age(self, age, at_upper_edge=False):
        """ref 기준 만 age세가 되는 생일. at_upper_edge=True면 그 나이의 끝(곧 생일)."""
        y = self.ref.year - age
        if at_upper_edge:
            return date(y - 1, self.ref.month, self.ref.day) + timedelta(days=1)
        d = date(y, self.rng.randint(1, 12), self.rng.randint(1, 28))
        return d if age_on(d, self.ref) == age else date(y - 1, d.month, d.day)

    @staticmethod
    def normalize(p):
        """현실성 보정: 만 19세 전 개업 금지, 2026년 개업 사업자는 2025년 매출 없음(0원)."""
        if p.get("opened") and p.get("birth"):
            earliest = date(p["birth"].year + 19, p["birth"].month, min(p["birth"].day, 28))
            if p["opened"] < earliest:
                p["opened"] = earliest
                p["biz_months"] = months_between(earliest, REF[0])
        if p.get("opened") and p["opened"].year >= 2026 and p.get("revenue") is not None:
            p["revenue"] = 0
        return p

    def set_months(self, p, m):
        p["opened"] = minus_months(self.ref, m)
        p["biz_months"] = months_between(p["opened"], self.ref)

    def set_opened(self, p, d):
        p["opened"] = d
        p["biz_months"] = months_between(d, self.ref)

    # ---- 후보 프로필 (유형, 대상 조건, 프로필) ----
    def candidates(self):
        n, r, rng = self.n, self.r, self.rng
        out = [("typical", "-", self.base()) for _ in range(6)]
        if "예비창업자" in r["types"]:
            out.append(("typical", "예비창업자", self.base("예비창업자")))

        def v(kind, target, fn, size=None):
            p = self.base(size)
            if p["size"] == "예비창업자" and target.startswith(("biz_months", "founded", "revenue")):
                p = self.base(next(s for s in r["types"] if s != "예비창업자"))
            fn(p)
            out.append((kind, target, self.normalize(p)))

        # 기업규모 불일치: 허용 안 되는 규모 중 가장 가까운 것
        bad = [s for s in SIZES if s not in r["types"]]
        for s in bad:
            def f(p, s=s):
                p["size"] = s
                if s == "예비창업자":
                    p["opened"] = p["biz_months"] = p["revenue"] = p["biz_type"] = None
                elif p.get("revenue") is None:
                    self.set_months(p, 24); p["revenue"] = REV_RANGE[s][0]; p["biz_type"] = "법인"
                else:
                    p["revenue"] = round(rng.uniform(*REV_RANGE[s]), -6)
                    if s != "소상공인":
                        p["biz_type"] = "법인"
            if s == "소상공인" and r.get("types_if_manufacturing"):
                def f(p):  # 16번: 제조업이 아닌 소상공인만 불가
                    p["size"] = "소상공인"; p["ksic"], p["ksic_name"] = GENERAL[0]
                    p["revenue"] = round(rng.uniform(*REV_RANGE["소상공인"]), -6)
            v("mismatch", "기업규모", f)

        a = n.get("age")
        if a:
            if a.get("max") is not None and not a.get("birth_year_min"):
                v("boundary_in", "age", lambda p: p.update(birth=self.birth_for_age(a["max"], True)))
                v("boundary_out", "age", lambda p: p.update(birth=self.birth_for_age(a["max"] + 1)))
            if a.get("min") is not None:
                v("boundary_in", "age", lambda p: p.update(birth=self.birth_for_age(a["min"])))
                v("boundary_out", "age", lambda p: p.update(birth=self.birth_for_age(a["min"] - 1, True)))
            if a.get("birth_year_min"):
                y = a["birth_year_min"]
                v("boundary_in", "birth_year", lambda p: p.update(birth=date(y, 1, 1)))
                v("boundary_out", "birth_year", lambda p: p.update(birth=date(y - 1, 12, 31)))

        reg = (n.get("region") or {}).get("allowed")
        if reg:
            for_gu = [a_ for a_ in reg if len(a_.split()) > 1]
            if for_gu:   # 같은 시 다른 구 (더 헷갈리는 경우)
                sido, gu = for_gu[0].split()
                v("mismatch", "region", lambda p: p.update(sido=sido, gu=rng.choice([g for g in GU[sido] if g != gu])))
            others = [s for s in SIDO if s not in {a_.split()[0] for a_ in reg}]
            def f(p):
                p["sido"] = rng.choice(others); p["gu"] = rng.choice(GU[p["sido"]])
            v("mismatch", "region", f)

        if r.get("allowed_ksic"):
            outside = [x for x in OUTSIDE if not any(x[0].startswith(a_) for a_ in r["allowed_ksic"])]
            v("mismatch", "industry", lambda p: p.update(zip(("ksic", "ksic_name"), rng.choice(outside))))
        if r.get("excl_table"):
            v("mismatch", "industry", lambda p: p.update(zip(("ksic", "ksic_name"), rng.choice(EXCLUDED))))

        b = n.get("biz_months")
        if b:
            if b.get("max") is not None:
                v("boundary_in", "biz_months", lambda p: self.set_months(p, b["max"]))
                if b.get("max_신산업"):
                    v("boundary_in", "biz_months", lambda p: (self.set_months(p, b["max_신산업"]), p["answers"].update(신산업_분야=True)))
                    v("boundary_out", "biz_months", lambda p: (self.set_months(p, b["max_신산업"] + 1), p["answers"].update(신산업_분야=True)))
                    v("other_req", "biz_months(신산업)", lambda p: (self.set_months(p, 100), p["answers"].update(신산업_분야=False)))
                elif b.get("max_면제"):
                    v("boundary_out", "biz_months", lambda p: (self.set_months(p, b["max"] + 1), p["answers"].update(업력면제_대상=False)))
                    v("typical", "biz_months(면제)", lambda p: (self.set_months(p, 100), p["answers"].update(업력면제_대상=True)))
                else:
                    v("boundary_out", "biz_months", lambda p: self.set_months(p, b["max"] + 1))
            if b.get("min"):
                v("boundary_in", "biz_months", lambda p: self.set_months(p, b["min"]))
                v("boundary_out", "biz_months", lambda p: self.set_months(p, b["min"] - 1))
            if b.get("founded_after"):
                d = date.fromisoformat(b["founded_after"])
                v("boundary_in", "founded_after", lambda p: self.set_opened(p, d))
                v("boundary_out", "founded_after", lambda p: self.set_opened(p, d - timedelta(days=1)))
            if b.get("founded_before"):
                d = date.fromisoformat(b["founded_before"])
                v("boundary_in", "founded_before", lambda p: self.set_opened(p, d))
                v("boundary_out", "founded_before", lambda p: self.set_opened(p, d + timedelta(days=1)))

        rv = n.get("revenue") or {}
        if rv.get("max") is not None:
            v("boundary_in", "revenue", lambda p: p.update(revenue=rv["max"]))
            v("boundary_out", "revenue", lambda p: p.update(revenue=rv["max"] + 1))
        if rv.get("min") is not None:
            v("boundary_in", "revenue", lambda p: p.update(revenue=rv["min"]))
            v("boundary_out", "revenue", lambda p: p.update(revenue=rv["min"] - 1))

        cond_ex = []
        for ex in n.get("exclusions") or []:
            f = ex["flag"]
            if f in INDUSTRY_FLAGS or f in SIZE_FLAGS:
                continue
            if "조건부" in ex.get("_메모", ""):
                cond_ex.append(f)
                continue
            v("exclusion", "제외:" + f, lambda p, f=f: (p["flags"].append(f), p.update(FLAG_KSIC.get(f, {}))))

        comp = set(r.get("computed", []))
        cr = r.get("credit", {})
        for req in n.get("other_requirements") or []:
            key = req["key"]
            if key not in comp:
                v("other_req", "기타:" + key, lambda p, key=key: p["answers"].update({key: False}))
            elif key == "신용점수":
                v("boundary_in", "기타:신용점수", lambda p: p.update(nice=cr["nice_min"], kcb=cr["kcb_min"] - 60))
                v("boundary_out", "기타:신용점수", lambda p: p.update(nice=cr["nice_min"] - 1, kcb=cr["kcb_min"] - 1))
            elif key == "저신용_또는_저소득":
                v("boundary_in", "기타:" + key, lambda p: p.update(nice=cr["nice_max"], kcb=cr["kcb_max"] + 80))
                v("boundary_out", "기타:" + key, lambda p: p.update(nice=cr["nice_max"] + 1, kcb=cr["kcb_max"] + 1, 저소득=False))
            elif key == "개인사업자":
                v("other_req", "기타:" + key, lambda p: p.update(biz_type="법인"))
            elif key == "여성기업":
                v("other_req", "기타:" + key, lambda p: p.update(gender="남"))
            elif key == "트랙_해당":
                v("other_req", "기타:" + key, lambda p: (self.set_months(p, 2), p.update(zip(("ksic", "ksic_name"), GENERAL[0]))))
            # 사업자등록_보유는 기업규모(예비창업자 불가)와 같은 조건이라 따로 안 만듦

        # 정보 부족 -> 조건부 후보
        miss = []
        def m(target, fn):
            p = self.base(next((s for s in r["types"] if s != "예비창업자"), None)); fn(p); miss.append(("missing", target, self.normalize(p)))
        if n.get("age"):
            m("age", lambda p: p.update(birth=None))
        if b and (b.get("min") or b.get("max") is not None):
            m("biz_months", lambda p: p.update(biz_months=None, opened=None))
        if rv.get("max") is not None:
            m("revenue", lambda p: p.update(revenue=None))
        if "신용점수" in comp:
            m("기타:신용점수", lambda p: p.update(nice=None, kcb=None))
        if b and b.get("max_신산업"):
            m("biz_months(신산업)", lambda p: self.set_months(p, 100))
        if b and b.get("max_면제"):
            m("biz_months(면제)", lambda p: self.set_months(p, 100))
        for f in cond_ex:
            m("제외:" + f, lambda p, f=f: p["flags"].append(f))
        plain = [q["key"] for q in n.get("other_requirements") or [] if q["key"] not in comp]
        if plain:
            m("기타:" + plain[0], lambda p, k=plain[0]: p["answers"].pop(k))
        return out, miss

    def pairs(self, k, want_cond):
        rng = self.rng
        cands, miss = self.candidates()
        lab = lambda xs: [(kind, tgt, p, *self.judge(p)) for kind, tgt, p in xs]
        allc = lab(cands)
        OK_KINDS = ("typical", "boundary_in")          # 가능을 의도한 유형
        pos = [x for x in allc if x[3] == POSSIBLE and x[0] in OK_KINDS]
        # 불가: 불가를 의도한 유형 중 조건 하나만 어긴 것만
        neg = [x for x in allc if x[3] == IMPOSSIBLE and len(x[4]) == 1 and x[0] not in OK_KINDS]
        cond = [x for x in lab(miss) if x[3] == CONDITIONAL]
        bad = [x for x in allc if x[3] == CONDITIONAL]          # 후보 중 의도치 않은 조건부는 버림
        rng.shuffle(cond)
        n_cond = 1 if (want_cond and cond) else 0
        # 불가: 서로 다른 대상 조건을 먼저, 경계값 우선
        neg.sort(key=lambda x: (not x[0].startswith("boundary"), rng.random()))
        seen, uniq, rest = set(), [], []
        for x in neg:
            (rest if x[1] in seen else uniq).append(x); seen.add(x[1])
        n_neg = k // 2
        neg_pick = (uniq + rest)[:n_neg]
        while len(neg_pick) < n_neg and neg:              # 불가 후보가 모자라면 같은 조건으로 다시 만듦
            c2, _ = self.candidates()
            more = [x for x in lab(c2) if x[3] == IMPOSSIBLE and len(x[4]) == 1 and x[0] not in OK_KINDS]
            if not more:
                break
            neg_pick.append(rng.choice(more))
        pos.sort(key=lambda x: (not (x[0].startswith("boundary") or x[1] != "-"), rng.random()))
        n_pos = k - len(neg_pick) - n_cond
        pos_pick = pos[:n_pos]
        while len(pos_pick) < n_pos:
            p = self.base()
            g = self.judge(p)
            if g[0] == POSSIBLE:
                pos_pick.append(("typical", "-", p, *g))
        chosen = neg_pick + cond[:n_cond] + pos_pick
        rng.shuffle(chosen)
        return chosen, len(bad)


# ---------------- 출력 ----------------
def won(v):
    """정확한 금액을 한국식으로 (경계값이 반올림되지 않게)."""
    if v is None:
        return "모름"
    v = int(v)
    if v == 0:
        return "0원"
    eok, rest = divmod(v, 100_000_000)
    man, one = divmod(rest, 10_000)
    parts = ([f"{eok:,}억"] if eok else []) + ([f"{man:,}만"] if man else []) + ([f"{one:,}"] if one else [])
    return " ".join(parts) + "원"


def profile_text(p, n, rule, ref):
    """시스템(챗봇)에 넣을 자연어 프로필."""
    s = []
    if p.get("birth"):
        s.append(f"대표자: {p['birth'].year}년 {p['birth'].month}월 {p['birth'].day}일생 (만 {age_on(p['birth'], ref)}세), {p['gender']}성")
    else:
        s.append(f"대표자: 생년월일 모름, {p['gender']}성")
    s.append(f"사업장(예정지) 소재지: {p['sido']} {p['gu']}")
    s.append(f"업종: {p['ksic_name']} (KSIC {p['ksic'][1:]})")
    if p["size"] == "예비창업자":
        s.append("사업자등록 전 예비창업자")
    else:
        s.append(f"기업 규모: {p['size']}, {p['biz_type']}사업자")
        s.append("개업일: 모름" if p.get("opened") is None else f"개업일: {p['opened']} (업력 {p['biz_months']}개월)")
        if p.get("opened") and p["opened"].year >= 2026:
            s.append("2025년 연매출: 없음 (2026년 개업)")
        else:
            s.append(f"2025년 연매출: {won(p.get('revenue'))}")
    if p.get("nice") is None and p.get("kcb") is None:
        s.append("신용점수: 모름")
    else:
        s.append(f"신용점수: NICE {p['nice']}점, KCB {p['kcb']}점")
    if rule.get("credit", {}).get("nice_max"):
        s.append("기초생활수급자·차상위계층: " + ("해당" if p.get("저소득") else "해당 없음"))
    # 신청 제외 사유는 짧은 이름(대표자_아님) 대신 정답표에 옮겨 둔 공고 원문 문장으로 적는다
    ev = {e["flag"]: e.get("evidence") or e["flag"].replace("_", " ") for e in n.get("exclusions") or []}
    if p["flags"]:
        s.append("신청 제외 사유: 공고의 다음 제외 사유에 해당함")
        s += [f"  - {ev.get(f, f.replace('_', ' '))}" for f in p["flags"]]
        s.append("그 밖의 신청 제외 사유: 해당하는 것 없음")
    else:
        s.append("신청 제외 사유: 해당하는 것 없음")
    qs = {q["key"]: q["question"] for q in n.get("other_requirements") or []}
    qs["신산업_분야"] = "창업지원법상 신산업 창업 분야(27개)에 해당하나요?"
    qs["업력면제_대상"] = "2026년 창업도시 조성 프로젝트 자금지원 수혜기업인가요?"
    comp = set(rule.get("computed", []))
    for key, q in qs.items():
        if key in comp:
            continue
        if key in p["answers"]:
            s.append(f"Q. {q} -> {'예' if p['answers'][key] else '아니오'}")
        elif key in {r_["key"] for r_ in n.get("other_requirements") or []}:
            s.append(f"Q. {q} -> 모름")
    return "\n".join(s)


def load_keys(d):
    keys = {}
    for path in sorted(glob.glob(os.path.join(d, "answer_key_*.json"))):
        m = re.search(r"answer_key_(\d\d)(_draft)?\.json$", path)
        if not m:
            continue
        num, draft = m.group(1), bool(m.group(2))
        if num in keys and draft:
            continue                     # 확정본이 있으면 초안은 무시
        keys[num] = (json.load(open(path, encoding="utf-8")), draft)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="data/answer_key")
    ap.add_argument("--rules", default="gen_rules.json")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--per-notice", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--max-cond", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rules = json.load(open(a.rules, encoding="utf-8"))
    ref = date.fromisoformat(rules["ref_date"])
    REF[0] = ref
    keys = load_keys(a.keys)
    drafts = [k for k, (_, d) in keys.items() if d]
    missing = [k for k in rules["notices"] if k not in keys]
    if drafts:
        print(f"주의: 확정본 없이 초안을 쓴 공고 {drafts}")
    if missing:
        print(f"주의: 정답표가 없는 공고 {missing}")

    rng = random.Random(a.seed)
    nums = sorted(keys)
    dev_only = set(rules.get("dev_only", []))
    pool = [k for k in nums if keys[k][0]["notice_id"] not in dev_only]
    rng.shuffle(pool)
    hold = set(pool[:round(len(nums) * a.holdout)])

    # 조건부는 전체의 max_cond보다 조금 낮게: 공고를 무작위로 골라 1개씩
    budget = int(len(nums) * a.per_notice * a.max_cond * 0.8)
    order = nums[:]; rng.shuffle(order)
    cond_set = set(order[:budget])

    fields = ["pair_id", "split", "순번", "notice_id", "case_type", "target_condition",
              "size", "birth", "age", "gender", "sido", "gu", "ksic", "ksic_name", "opened", "biz_months",
              "revenue", "biz_type", "nice", "kcb", "flags", "answers", "profile_text", "gold", "gold_reason"]
    rows, dropped = [], 0
    for num in nums:
        n, _ = keys[num]
        rule = rules["notices"][num]
        if rule.get("biz_months_from_legal"):      # 법률 용어(창업기업 등)에서 나오는 업력 상한을 숫자로 보충
            n = {**n, "biz_months": {**(n.get("biz_months") or {"min": None}), **rule["biz_months_from_legal"]}}
        mk = Maker(n, rule, ref, rng)
        chosen, bad = mk.pairs(a.per_notice, num in cond_set)
        dropped += bad
        split = "holdout" if num in hold else "dev"
        for i, (kind, tgt, p, gold, why) in enumerate(chosen, 1):
            rows.append({
                "pair_id": f"{num}-P{i:02d}", "split": split, "순번": int(num), "notice_id": n["notice_id"],
                "case_type": kind, "target_condition": tgt, "size": p["size"],
                "birth": p.get("birth") or "", "age": age_on(p["birth"], ref) if p.get("birth") else "",
                "gender": p.get("gender") or "", "sido": p["sido"], "gu": p["gu"],
                "ksic": p["ksic"], "ksic_name": p["ksic_name"],
                "opened": p.get("opened") or "", "biz_months": "" if p.get("biz_months") is None else p["biz_months"],
                "revenue": "" if p.get("revenue") is None else int(p["revenue"]), "biz_type": p.get("biz_type") or "",
                "nice": "" if p.get("nice") is None else p["nice"], "kcb": "" if p.get("kcb") is None else p["kcb"],
                "flags": "|".join(p["flags"]),
                "answers": "|".join(f"{k}={'예' if v else '아니오'}" for k, v in p["answers"].items()),
                "profile_text": profile_text(p, n, rule, ref), "gold": gold, "gold_reason": "|".join(why),
            })

    os.makedirs(a.out, exist_ok=True)
    for name, rs in (("all", rows), ("dev", [r for r in rows if r["split"] == "dev"]),
                     ("holdout", [r for r in rows if r["split"] == "holdout"])):
        with open(os.path.join(a.out, f"pairs_{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)
    rep = report(rows, a.max_cond, keys, hold)
    open(os.path.join(a.out, "gold_report.md"), "w", encoding="utf-8").write(rep)
    print(rep)
    print(f"\n저장: {a.out}/pairs_all.csv, pairs_dev.csv, pairs_holdout.csv, gold_report.md")


def report(rows, max_cond, keys, hold):
    L = []
    pct = lambda x, t: f"{x} ({x / t:.0%})" if t else "0"
    L.append("## 정답 분포 보고 (평가 전)\n")
    L.append("| 구분 | 공고 | 쌍 | 신청 가능 | 신청 불가 | 조건부 | 경계값 쌍 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, rs in (("개발셋", [r for r in rows if r["split"] == "dev"]),
                     ("홀드아웃", [r for r in rows if r["split"] == "holdout"]), ("합계", rows)):
        t = len(rs); c = lambda g: sum(r["gold"] == g for r in rs)
        L.append(f"| {name} | {len({r['notice_id'] for r in rs})} | {t} | {pct(c(POSSIBLE), t)} | {pct(c(IMPOSSIBLE), t)} | "
                 f"{pct(c(CONDITIONAL), t)} | {sum(r['case_type'].startswith('boundary') for r in rs)} |")
    tc = sum(r["gold"] == CONDITIONAL for r in rows) / len(rows)
    L.append(f"\n전체 조건부 비율 {tc:.1%} (기준 {max_cond:.0%} 이하) - {'통과' if tc <= max_cond else '초과'}")
    L.append(f"홀드아웃 공고: {', '.join(sorted(hold))}")

    L.append("\n### 불가 쌍이 어떤 조건에서 나왔나 (대상 조건별)\n")
    L.append("| 대상 조건 | 불가 쌍 | 그중 경계값 |")
    L.append("|---|---|---|")
    from collections import Counter
    grp = lambda t: t.split(":")[0] if ":" in t else t
    cn = Counter(grp(r["target_condition"]) for r in rows if r["gold"] == IMPOSSIBLE)
    cb = Counter(grp(r["target_condition"]) for r in rows if r["gold"] == IMPOSSIBLE and r["case_type"].startswith("boundary"))
    for k_, v_ in cn.most_common():
        L.append(f"| {k_} | {v_} | {cb[k_]} |")

    L.append("\n### 공고별\n")
    L.append("| 순번 | 분할 | 가능 | 불가 | 조건부 | 경계값 |")
    L.append("|---|---|---|---|---|---|")
    warn = []
    for num in sorted(keys):
        rs = [r for r in rows if r["순번"] == int(num)]
        c = lambda g: sum(r["gold"] == g for r in rs)
        L.append(f"| {num} | {rs[0]['split']} | {c(POSSIBLE)} | {c(IMPOSSIBLE)} | {c(CONDITIONAL)} | "
                 f"{sum(r['case_type'].startswith('boundary') for r in rs)} |")
        if c(IMPOSSIBLE) < 3 or c(POSSIBLE) < 3:
            warn.append(f"- {num}: 가능 {c(POSSIBLE)} / 불가 {c(IMPOSSIBLE)}")
    L.append("\n점검 필요 공고: " + ("없음" if not warn else "\n" + "\n".join(warn)))
    return "\n".join(L)


if __name__ == "__main__":
    main()
