"""v2 장면 분석 확인 · 사진 1장 = 결과 1개 (조각 확률 평균) 규칙을 Validation 과 밭 사진으로 확인
실행: ~/fastapi_resnet18 에서  python tools/v2_check.py [외부사진폴더]
  - 서비스와 같은 코드 (ai/scene_service.classify_scene) · 같은 최종 모델 (.env 의 MODEL_PATH, 없으면 full_aug)
  - Validation 133장 : v1 (사진 전체) 과 v2 (장면 분석) 의 결과가 같은지 · 정확도 · 경로 · 판정
  - 밭 사진 : 저장소 robust/variants/00_original.jpg + 외부사진 폴더
출력: reports/data/v2_check.json
규칙 : Test 는 사용하지 않음 (기준 확인은 Validation 으로만)
"""
import json, os, sys, types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
os.environ.setdefault("MODEL_PATH", "artifacts/beans_resnet18_full_aug")

from ai import services                                   # noqa: E402
from ai.scene_service import classify_scene               # noqa: E402

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
EXT = {".jpg", ".jpeg", ".png", ".webp"}
val = [(c, p) for c in CLASSES for p in sorted(Path("dataset/val", c).glob("*")) if p.suffix.lower() in EXT]

clf = services.get_classifier()
rows, modes, decisions = [], Counter(), Counter()
v1_ok = v2_ok = same = 0
for c, p in val:
    v1 = max(clf(services.load_rgb_image(p.read_bytes()), top_k=3), key=lambda x: x["score"])["label"]
    r = classify_scene(p.read_bytes())
    modes[r["mode"]] += 1; decisions[r["decision"]] += 1
    v1_ok += v1 == c; v2_ok += r["label"] == c; same += v1 == r["label"]
    rows.append({"file": f"{c}/{p.name}", "true": c, "v1": v1, "v2_label": r["label"], "v2_score": r["score"],
                 "mode": r["mode"], "parts": r["parts"], "decision": r["decision"]})
n = len(val)
summary = {"model": os.environ["MODEL_PATH"], "val_count": n, "modes": dict(modes), "decisions": dict(decisions),
           "v1_accuracy": round(v1_ok / n, 4), "v2_accuracy": round(v2_ok / n, 4), "v1_v2_same": same,
           "v2_changed": [x for x in rows if x["v1"] != x["v2_label"]]}
print(f"모델 {summary['model']}")
print(f"Validation {n}장 · 경로 {dict(modes)} · 판정 {dict(decisions)}")
print(f"  v1 사진 전체 정확도 {summary['v1_accuracy']} · v2 정확도 {summary['v2_accuracy']} · 같은 결과 {same}/{n}")
for x in summary["v2_changed"]:
    print(f"  달라진 사진 {x['file']} · v1 {x['v1']} → v2 {x['v2_label']} {x['v2_score']} ({x['mode']} · {x['parts']}조각)")

field = [Path("robust/variants/00_original.jpg")]
ext = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/c/Users/user1/Downloads/교과9_평가/외부사진")
if ext.exists():
    field += sorted(p for p in ext.glob("*") if p.suffix.lower() in EXT)
summary["field"] = []
print("\n밭 · 외부 사진")
for p in field:
    if not p.exists():
        continue
    r = classify_scene(p.read_bytes())
    summary["field"].append({"file": p.name, "mode": r["mode"], "parts": r["parts"], "decision": r["decision"],
                             "label": r["label"], "score": r["score"], "all": r["all"]})
    probs = " · ".join(f"{a['label']} {a['score']:.2f}" for a in r["all"])
    print(f"  {p.name:22s} {r['mode']:6s} 조각 {r['parts']} · 판정 {r['decision']:8s} · 평균 {probs}")

Path("reports/data").mkdir(parents=True, exist_ok=True)
Path("reports/data/v2_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n저장: reports/data/v2_check.json")
