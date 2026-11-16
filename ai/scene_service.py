"""v2 장면 분석 서비스 · 사진 1장 → 품질 게이트 → 근접/원거리 판별 → 분류 → **사진 1장에 결과 1개**

    분류(Classification) 프로젝트이므로 결과는 사진 전체에 대해 라벨 1개 + 점수 1개 (박스 · 번호 표시 없음).
    close (근접 · 잎 1장) : 사진 전체를 그대로 공통 모델에 (v1 과 같은 입력)
    far   (원거리 · 밭)    : 학습 사진처럼 크게 보이도록 사진 일부를 잘라 공통 모델에 넣고,
                            조각들의 3클래스 확률을 **평균**해 사진 전체 결과 1개로 합침 (multi-crop 평균)
    판정:
        품질 불합격 → RETAKE · 잎 없음 → NO_LEAF
        평균 1등 점수 < SCENE_CHECK_SCORE → CHECK (사람 확인)
        평균 1등이 healthy → HEALTHY · 병해 → DISEASE
"""
import base64
import time

import cv2
import numpy as np

from . import services
from .config import MODEL_PATH, SCENE_CHECK_SCORE
from .leaf_router import analyze_scene, check_quality, draw_result

HEALTHY = "healthy"


def _predict(images):
    """공통 모델 (v1 과 같은 get_classifier) 로 여러 장을 한 번에 분류 → 조각마다 {라벨: 확률}."""
    raw = services.get_classifier()(images, top_k=3)
    if images and isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = [raw]                                     # 1장일 때 리스트가 한 겹 덜 감싸진 경우
    return [{str(x["label"]): float(x["score"]) for x in item} for item in raw]


def _average(probs):
    """조각들의 클래스별 확률 평균 → 큰 순서 [{label, score}]."""
    labels = sorted({k for p in probs for k in p})
    mean = {k: float(np.mean([p.get(k, 0.0) for p in probs])) for k in labels}
    return sorted(({"label": k, "score": round(v, 4)} for k, v in mean.items()), key=lambda x: x["score"], reverse=True)


def _decide(top_label: str, top_score: float) -> str:
    if top_score < SCENE_CHECK_SCORE:
        return "CHECK"
    return "HEALTHY" if top_label == HEALTHY else "DISEASE"


def classify_scene(image_bytes: bytes, draw: bool = False) -> dict:
    image = services.load_rgb_image(image_bytes)        # v1 과 같은 검증 (형식 · 크기)
    rgb = np.asarray(image)
    started = time.perf_counter()
    quality = check_quality(rgb)
    scene = analyze_scene(rgb)
    s = scene.to_dict()
    result = {"model_id": MODEL_PATH, "quality": quality.__dict__,
              "scene": {k: s[k] for k in ("mode", "leaf_ratio", "largest_hull", "similar", "whole_leaves")},
              "mode": scene.mode, "parts": 0, "label": None, "score": None, "all": [],
              "decision": None, "check": False, "inference_ms": None}

    if not quality.ok:
        result["decision"] = "RETAKE"
    elif scene.mode == "no_leaf":
        result["decision"] = "NO_LEAF"
    else:
        crops = [image] if scene.mode == "close" else [image.crop(b) for b in scene.boxes]
        t0 = time.perf_counter()
        probs = _predict(crops)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        avg = _average(probs)
        top = avg[0]
        result.update(parts=len(crops), label=top["label"], score=top["score"], all=avg,
                      decision=_decide(top["label"], top["score"]), check=top["score"] < SCENE_CHECK_SCORE)

    result["total_ms"] = round((time.perf_counter() - started) * 1000, 2)

    if draw:
        canvas = draw_result(rgb, result)
        if max(canvas.shape[:2]) > 900:
            k = 900 / max(canvas.shape[:2])
            canvas = cv2.resize(canvas, (round(canvas.shape[1] * k), round(canvas.shape[0] * k)))
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["image_jpg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    return result
