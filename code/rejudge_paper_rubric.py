
"""Apply a continuous rubric to existing SocialMemBench answers and report mean question and mean network scores.
The script reads questions through the benchmark loader and scores answer records already stored under the result directory.

Example: python3 rejudge_paper_rubric.py --baseline hipporag
"""
from __future__ import annotations
import os, sys, json, glob, argparse, re
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"8")
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"GroupMemBench"))
from baselines.rag_common.eval_lib import load_env_file
load_env_file(str(HERE/".env"))
from bench_loaders import iter_units

JUDGE_SYS = ("You are an expert grader evaluating AI answers about social group conversations.\n"
             "You score answers on a 0.0-1.0 scale based on factual correctness only.")
JUDGE_USER = """Score the generated answer against the gold answer on a 0.0-1.0 scale.

CRITICAL RULES (apply before all others):
1. A concise correct answer scores THE SAME as a verbose one. Do NOT penalize
   brevity or reward evidence citation. If the core fact is correct, score 1.0.
2. Attributing a fact to the wrong person is always a critical error -> score <= 0.3,
   regardless of how much other content is correct.
3. "NOT ENOUGH INFORMATION" or "[No memories retrieved]" always scores 0.0.

General rubric:
- 1.0: Core fact correct, correct person/attribution, required parts present
- 0.7-0.9: Right answer but one minor part missing or slightly imprecise
- 0.4-0.6: Right direction but wrong detail, missing key element, or ambiguous
- 0.0-0.3: Wrong attribution, wrong fact, contradiction, or no answer

Q-type specific rules:
- Q1/Q3: Core fact + correct person = 1.0. For Q3, score = fraction of group members correctly recalled.
- Q4 (attribution): Correct speaker named AND foil NOT named -> 1.0. Correct + names foil -> 0.7. Wrong -> 0.0-0.3.
- Q5 (theory of mind): BOTH parts required for 1.0 (preference + who revealed it with observable action). One part -> 0.5.
- Q8 (temporal shift): old state(+0.33)+new state(+0.33)+trigger(+0.33). Both states no trigger -> 0.7. One state -> 0.4.

Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

Output JSON only - no preamble:
{{"score": 0.0, "rationale": "one sentence"}}"""

def parse_score(content: str) -> float:
    try:
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        return max(0.0, min(1.0, float(json.loads(content).get("score", 0.0))))
    except Exception:
        m = re.search(r'\b(0\.\d+|1\.0|0|1)\b', content)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit-per-net", type=int, default=0, help=">0: score only the first N questions in each category for each network")
    ap.add_argument("--results-dir", default="results", help="root directory containing benchmark result files")
    ap.add_argument("--tag", default="", help="output filename suffix")
    args = ap.parse_args()


    qtext = {}
    for unit, msgs, qs in iter_units("socialmem", HERE):
        for q in qs: qtext[q["id"]] = (q["question"], q.get("category","?"))


    rows = []
    for f in glob.glob(f"{args.results_dir}/socialmem/*/{args.baseline}__*.jsonl"):
        net = f.split("/")[-2]; cat = f.split(f"{args.baseline}__")[1][:-6]
        recs = [json.loads(l) for l in open(f)]
        if args.limit_per_net: recs = recs[:args.limit_per_net]
        for r in recs:
            qid = r.get("id",""); qq = qtext.get(qid,("",cat))
            rows.append((net, cat, qid, qq[0], r.get("gold",""), r.get("agent_answer","")))

    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    def judge(row):
        net,cat,qid,question,gold,ans = row
        if cat=="Q4" or cat=="Q5":
            pass
        up = JUDGE_USER.format(question=question, golden_answer=gold, generated_answer=ans)
        try:
            r = cli.chat.completions.create(model=args.judge_model,
                messages=[{"role":"system","content":JUDGE_SYS},{"role":"user","content":up}],
                temperature=0, max_tokens=100000)
            return (net,cat,parse_score(r.choices[0].message.content or ""))
        except Exception as e:
            return (net,cat,0.0)
    print(f"[{args.baseline}] scoring {len(rows)} questions with the continuous rubric (judge={args.judge_model})...")
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(judge, rows))

    alls = [s for _,_,s in out]
    meanq = float(np.mean(alls))
    bynet = {}; bycat = {}
    for net,cat,s in out:
        bynet.setdefault(net,[]).append(s); bycat.setdefault(cat,[]).append(s)
    meann = float(np.mean([np.mean(v) for v in bynet.values()]))
    print(f"\n==== {args.baseline} continuous-rubric scores ====")
    print(f"  MeanQ = {meanq:.3f}   MeanN = {meann:.3f}   networks={len(bynet)} questions={len(alls)}")
    print(f"  fraction >=0.7 = {np.mean([s>=0.7 for s in alls])*100:.1f}%")
    print("  MeanQ by category:")
    for cat in sorted(bycat):
        v = bycat[cat]; print(f"    {cat}: {np.mean(v):.3f}  (n={len(v)})")
    Path("results").mkdir(exist_ok=True)
    json.dump({"baseline":args.baseline,"meanq":meanq,"meann":meann,
               "by_cat":{c:float(np.mean(v)) for c,v in bycat.items()},
               "ge07_rate":float(np.mean([s>=0.7 for s in alls]))},
              open(f"results/papermetric_socialmem_{args.baseline}{args.tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()
