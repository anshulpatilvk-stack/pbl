import io
import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# 1. PREPROCESSING + FRUIT SEGMENTATION
#    Biggest accuracy fix: only analyse FRUIT pixels, never the background.
# ---------------------------------------------------------------------------

def _load_image(image_bytes, max_side=320):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")      # fix phone rotation
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def _largest_blob(mask):
    mask = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask, dtype=bool)
    biggest = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 1, thickness=-1)
    return out.astype(bool)


def _segment_banana(img_rgb):
    """
    Find the banana by COLOUR (yellow / yellow-green skin) instead of guessing the
    background, so wood tables, counters, dark or coloured surfaces and shadows are
    excluded. Brown patches that are attached to the banana (sugar spots, bruises, tips)
    are pulled back in. Returns None if no banana-coloured region is found.
    """
    h, w = img_rgb.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    core = ((H >= 19) & (H <= 48) & (S >= 90) & (V >= 70)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cv2.contourArea(c) for c in cnts)
    keep = [c for c in cnts if cv2.contourArea(c) >= max(0.2 * biggest, 0.004 * total)]  # banana bunch
    if not keep or sum(cv2.contourArea(c) for c in keep) < 0.03 * total:
        return None                                   # not enough banana colour -> fallback
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, keep, -1, 1, thickness=-1)       # also fills spots inside

    # Pull back brown patches touching the banana (but never bg-sized regions like wood)
    brown = ((H >= 5) & (H <= 22) & (S >= 50) & (V >= 25) & (V < 170)).astype(np.uint8)
    near = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(brown & (1 - filled), connectivity=8)
    core_area = float(filled.sum())
    for i in range(1, n_lab):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = lab == i
        if area <= 0.35 * core_area and near[comp].any():
            filled[comp] = 1
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    area = float(filled.sum()) / total
    return filled.astype(bool) if 0.02 < area < 0.90 else None


def _segment_fruit(img_rgb, cat=None):
    """GrabCut first, background-colour distance as fallback, centre ellipse last."""
    h, w = img_rgb.shape[:2]
    total = h * w
    if cat == "banana":
        m = _segment_banana(img_rgb)
        if m is not None:
            return m
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- GrabCut ---
    try:
        gc = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        m = _largest_blob((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        if 0.08 * total < m.sum() < 0.92 * total:
            return m
    except Exception:
        pass

    # --- Fallback: pixels that differ from the border (background) colour ---
    border = np.concatenate([img_rgb[0], img_rgb[-1], img_rgb[:, 0], img_rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img_rgb.astype(float) - bg, axis=2)
    m = _largest_blob(dist > 45)
    if 0.08 * total < m.sum() < 0.92 * total:
        return m

    # --- Last resort: central ellipse ---
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.35), int(h * 0.35)), 0, 0, 360, 1, -1)
    return m.astype(bool)



# ---------------------------------------------------------------------------
# 2b. SURFACE QUALITY: image sharpness + wrinkle/crease detection
# ---------------------------------------------------------------------------

def _sharpness(img_rgb, mask):
    """
    Edge sharpness of the fruit outline = (edge gradient) / (edge contrast), measured in
    COLOUR right at the boundary (not whole-fruit brightness), so a banana on a wooden
    table or a dark red apple on a dark surface isn't mistaken for a blurry image.
    ~1/edge-width in pixels: sharp ~0.3+, blurry < ~0.15.
    """
    img = img_rgb.astype(np.float32)
    m = mask.astype(np.uint8)
    k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))   # ~8 px from the edge
    k_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))   # ~17 px from the edge
    ring = (cv2.dilate(m, k_edge) - cv2.erode(m, k_edge)).astype(bool)
    # contrast bands sit FAR enough from the edge that blur doesn't mix them together
    band_in = (cv2.erode(m, k_a) - cv2.erode(m, k_b)).astype(bool)
    band_out = (cv2.dilate(m, k_b) - cv2.dilate(m, k_a)).astype(bool)
    if band_in.sum() < 20:                                           # very thin fruit
        band_in = cv2.erode(m, k_edge).astype(bool) & ~ring
    if band_out.sum() < 20:
        band_out = cv2.dilate(m, k_b).astype(bool) & ~cv2.dilate(m, k_edge).astype(bool)
    if ring.sum() < 20 or band_in.sum() < 20 or band_out.sum() < 20:
        return None
    contrast = float(np.linalg.norm(img[band_in].mean(axis=0) - img[band_out].mean(axis=0)))
    sm = img
    g2 = np.zeros(mask.shape, np.float32)
    for c in range(3):
        gx = cv2.Sobel(sm[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(sm[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g2 += gx * gx + gy * gy
    grad = np.sqrt(g2)[ring]
    if contrast < 15:                       # fruit colour ~ background colour: can't judge
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return min(1.0, float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 400.0) * 0.3 + 0.1
    return float(np.percentile(grad, 90)) / contrast


def _wrinkle_metrics(img_rgb, mask):
    """
    Wrinkles/creases = many thin dark lines in the SAME colour as the skin (just darker).
    Colour streaks (Fuji/Gala stripes) and spots change hue, so they are ignored here.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inner = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    if inner.sum() < 100:
        inner = mask
    # Black top-hat keeps only THIN dark structures; shading / highlights / rim ignored.
    gray_s = cv2.GaussianBlur(gray, (0, 0), 0.8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thin_dark = cv2.morphologyEx(gray_s, cv2.MORPH_BLACKHAT, kern)

    # Hue difference between each pixel and the fruit's overall hue (circular).
    ang = hsv[..., 0].astype(np.float32) * (np.pi / 90.0)
    w = inner.astype(np.float32)
    mc = float((np.cos(ang) * w).sum() / max(w.sum(), 1))
    ms = float((np.sin(ang) * w).sum() / max(w.sum(), 1))
    ref = np.arctan2(ms, mc)
    dh = np.abs(np.angle(np.exp(1j * (ang - ref)))) * (90.0 / np.pi)       # OpenCV hue units
    same_colour = (dh < 5.0) | (hsv[..., 1] < 40)

    crease_px = (thin_dark > 10) & same_colour
    mean_g = float(gray[inner].mean()) + 1.0
    return {
        "crease": float(crease_px[inner].mean()),
        "detail": float((thin_dark * same_colour)[inner].mean() / mean_g),
    }


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION  (pixel FRACTIONS in HSV ranges, not channel averages)
#    OpenCV hue range is 0-179. Red wraps around (0-10 and 165-179), so a
#    plain mean of hue is meaningless for red fruit.
# ---------------------------------------------------------------------------

def _extract_features(img_rgb, mask, cat=None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = (hsv[..., i][mask].astype(int) for i in range(3))
    n = max(len(H), 1)

    red    = ((H <= 10) | (H >= 165)) & (S >= 90) & (V >= 50)
    orange = (H > 10) & (H < 20) & (S >= 90) & (V >= 60)
    yellow = (H >= 20) & (H <= 35) & (S >= 70) & (V >= 90)
    green  = (H > 35) & (H <= 85) & (S >= 50) & (V >= 40)
    brown  = (H >= 8) & (H <= 25) & (S >= 50) & (V >= 30) & (V < 110)
    dark   = V < 45
    gloss  = (S < 50) & (V > 200)

    f = lambda a: float(a.sum()) / n

    # --- Local (relative) spot detection, used by apple & banana ----------------
    # A real spot/bruise is darker than ITS OWN SURROUNDINGS, whereas natural shading,
    # deep-red skin and the shadowed side of a banana are dark everywhere. Comparing
    # each pixel with the local average removes those false positives.
    Vf = hsv[..., 2].astype(np.float32)
    rim = 25 if cat == "banana" else 11            # banana: ignore stem/end tips & rim shadow
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim, rim))
    inner = cv2.erode(mask.astype(np.uint8), k_in).astype(bool)      # ignore rim/shadow
    if inner.sum() < 100:
        inner = mask
    inner_f = inner.astype(np.float32)
    local_v = cv2.GaussianBlur(Vf * inner_f, (0, 0), 14) / (cv2.GaussianBlur(inner_f, (0, 0), 14) + 1e-6)
    spot_map = (Vf < SPOT_RATIO * local_v) & (Vf < 150) & inner
    Hf = hsv[..., 0].astype(int); Sf = hsv[..., 1].astype(int)
    rot_map = (Hf >= 8) & (Hf <= 20) & (Sf >= 70) & (Vf < 75) & inner   # dark brown
    brown_b_map = (Hf >= 8) & (Hf <= 19) & (Sf >= 50) & (Vf < 150) & inner  # banana browning
    # Brown patches on an apple whose own hue is NOT brown (bruises / rot, any size)
    ang = Hf.astype(np.float32) * (np.pi / 90.0)
    ref = np.arctan2(float((np.sin(ang) * inner_f).sum()), float((np.cos(ang) * inner_f).sum()))
    dh = np.abs(np.angle(np.exp(1j * (ang - ref)))) * (90.0 / np.pi)
    brown_anom_map = brown_b_map & (dh > 5.0)
    ref_deg = (float(np.degrees(ref)) % 360.0) / 2.0                 # banana's own hue (OpenCV units)
    banana_brown_map = None   # built below, once yellow_l / green_l exist
    n_in = max(int(inner.sum()), 1)
    g = lambda a: float(a.sum()) / n_in

    red_any   = ((H <= 10) | (H >= 165)) & (S >= 60)                     # any shade of red
    yellow_l  = (H >= 18) & (H <= 35) & (S >= 70) & (V >= 60)            # incl. shaded yellow
    green_l   = (H > 35) & (H <= 85) & (S >= 40) & (V >= 40)

    little_yg = (f(yellow_l) + f(green_l)) < 0.25                    # almost no yellow/green skin
    banana_all_brown = (ref_deg < 19.0) or little_yg                 # whole peel is brown/orange
    banana_brown_map = brown_b_map & ((dh > 6.0) | banana_all_brown)

    wr = _wrinkle_metrics(img_rgb, mask)
    sharp = _sharpness(img_rgb, mask)
    return {
        "crease": wr["crease"], "detail": wr["detail"],
        "sharpness": sharp if sharp is not None else 0.3,
        "red": f(red), "orange": f(orange), "yellow": f(yellow), "green": f(green),
        "brown": f(brown), "dark": f(dark), "defect": f(brown | dark),
        "gloss": f(gloss),
        "red_any": f(red_any), "yellow_l": f(yellow_l), "green_l": f(green_l),
        "spot": g(spot_map), "rot": g(rot_map),
        "defect_local": g(spot_map | rot_map | brown_anom_map),
        "banana_defect": g(spot_map | rot_map | banana_brown_map),
        "mean_v": float(V.mean()) if len(V) else 0.0,
        "mean_s": float(S.mean()) if len(S) else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. CATEGORY SCORING  (stage -> base score, then subtract defect penalties)
# ---------------------------------------------------------------------------

def _score_tomato(f):
    golden = f["orange"] + f["yellow"]            # golden / yellow-orange skin share
    turning = f["red"] + 0.5 * f["orange"]
    if f["red"] >= 0.60:
        base, stage = 93, "Peak Fresh (Ripe & Firm)"
        base -= min(GOLDEN_PENALTY_MAX, golden * 25)        # red with golden patches
    elif f["red"] < 0.25 and golden >= 0.40:
        base, stage = GOLDEN_BASE_SCORE, "Golden / Yellow-Orange (Not Red - Reduced Freshness)"
    elif turning >= 0.45:
        base, stage = 80, "Semi-Ripe (Turning Red)"
    elif f["green"] >= 0.40:
        base, stage = 68, "Unripe (Green)"
    else:
        base, stage = 55, "Poor Colour / Overripe"
    penalty = min(55, f["defect"] * 160)
    sev = _wrinkle_severity(f)
    if sev >= 0.35:
        stage = "Wrinkled / Dehydrated (Past Prime)"
    elif f["defect"] > 0.10 and f["mean_v"] < 100:
        stage = "Overripe / Spoiling"
    score = base - penalty + min(f["gloss"] * 40, 4) * (sev < 0.15)
    colour_match = f["red"] + f["orange"] + f["yellow"] + f["green"]
    return _apply_wrinkles(score, sev, 70), stage, colour_match


def _score_banana(f):
    """
    Ripening scale: green -> yellow-green (ripening) -> yellow (peak) -> sugar spots
    (very ripe, sweet) -> heavy browning (overripe). Spots are measured RELATIVE to the
    peel, so shadow / shaded side / stem tip no longer count as defects.
    """
    yellow, green, spots = f["yellow_l"], f["green_l"], f["banana_defect"]

    if spots >= 0.25:
        score = max(10.0, 48 - 100 * (spots - 0.25))
        stage = "Overripe (Heavily Browned / Bruised)"
    elif spots >= 0.12:
        score = 72 - 150 * (spots - 0.12)                 # 72 -> ~53
        stage = "Very Ripe (Many Sugar Spots - Eat / Bake Soon)"
    elif spots >= 0.04:
        score = 88 - 200 * (spots - 0.04)                 # 88 -> ~72
        stage = "Ripe with Sugar Spots (Sweet)"
    elif green >= 0.55 and yellow < 0.35:
        score = 68
        stage = "Unripe (Green)"
    elif green >= 0.08 and yellow >= 0.20:
        score = 93 - 45 * (green - 0.08)                  # 93 (just turning) -> ~71 (mostly green)
        stage = "Ripening (Yellow-Green)"
    elif yellow + green >= 0.35:
        score = 95
        stage = "Optimal Ripeness (Sweet & Fresh)"
    else:
        score = 62
        stage = "Dull / Past Peak"
    return score, stage, yellow + green


def _score_apple(f):
    """
    Colour-independent: red, yellowish-red, green and golden apples are all fine.
    Only real defects (bruises, dark spots, rot) and genuine shrivel lower the score.
    """
    colour = f["red_any"] + f["orange"] + f["yellow"] + f["green_l"]
    d = f["defect_local"]
    score = 95 - min(65.0, d * 260)
    if d < 0.02:
        stage = "Crisp & Fresh (Grade A)"
    elif d < 0.06:
        stage = "Fresh (Minor Blemishes)"
    elif d < 0.14:
        stage = "Moderate (Bruising / Spots)"
    elif d < 0.25:
        stage = "Poor (Bruised / Softening)"
    else:
        stage = "Spoiled (Rotting)"
    sev = _wrinkle_severity(f, APPLE_WRINKLE_CREASE, APPLE_WRINKLE_DETAIL)
    if sev >= 0.35:
        stage = "Shriveled / Wrinkled (Old)"
    if sev < 0.15 and d < 0.06:
        score += 2 if f["mean_s"] > 90 else 0
        score += min(f["gloss"] * 40, 3)
    return _apply_wrinkles(score, sev, 45, slope=60.0), stage, colour


def _score_lemon(f):
    colour = f["yellow"] + 0.5 * f["green"]
    if f["yellow"] >= 0.60 and f["defect"] < 0.05:
        base, stage = 93, "Juicy & Fresh (Citrus Peak)"
    elif f["yellow"] >= 0.35 or f["green"] >= 0.40:
        base, stage = 80, "Good (Slightly Green / Mixed)"
    else:
        base, stage = 58, "Dull / Ageing"
    if f["brown"] > 0.12:
        stage = "Ageing (Browning Skin)"
    penalty = min(45, f["defect"] * 130)
    sev = _wrinkle_severity(f)
    if sev >= 0.35:
        stage = "Dried / Wrinkled Skin"
    return _apply_wrinkles(base - penalty, sev, 50), stage, colour


# Wrinkle tuning: (onset, full) values. Below onset = smooth skin (no penalty);
# at/above full = severely wrinkled (score forced to <= ~35).
WRINKLE_CREASE = (0.015, 0.05)   # share of crease pixels
WRINKLE_DETAIL = (0.015, 0.025) # relative crease energy


SPOT_RATIO = 0.62               # pixel counts as a spot if < 62% of its local brightness
APPLE_WRINKLE_CREASE = (0.05, 0.14)   # apples have natural streaks/lenticels -> less sensitive
APPLE_WRINKLE_DETAIL = (0.030, 0.060)
WRINKLE_CAP_SLOPE = 75.0        # higher = harsher cap for wrinkled fruit
GOLDEN_BASE_SCORE = 70          # tomato that is golden/yellow-orange instead of red
GOLDEN_PENALTY_MAX = 10         # extra loss for red tomatoes with golden patches


def _wrinkle_severity(f, crease_rng=None, detail_rng=None):
    """0.0 = smooth skin ... 1.0 = severely wrinkled (uses the stronger of 2 signals)."""
    def ramp(x, lo_hi):
        lo, hi = lo_hi
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    sev = max(ramp(f["crease"], crease_rng or WRINKLE_CREASE),
              ramp(f["detail"], detail_rng or WRINKLE_DETAIL))
    return sev ** 0.7   # ramps up faster: moderate wrinkles already hurt a lot


def _apply_wrinkles(score, sev, max_pen, slope=None):
    """Subtract a penalty AND cap the score: sev 1.0 -> max 20, sev 0.5 -> max ~57."""
    score -= sev * max_pen
    return min(score, 95.0 - (slope or WRINKLE_CAP_SLOPE) * sev)


BLUR_THRESHOLD = 0.18   # edge sharpness below this = blurry (calibrate on your photos)
BLUR_SCORE_CAP = 70.0   # max score allowed for a blurry image

_SCORERS = {"tomato": _score_tomato, "banana": _score_banana,
            "apple": _score_apple, "lemon": _score_lemon}

# (min_score, shelf-life text) per category, checked top to bottom
_SHELF_LIFE = {
    "tomato": [(90, "5 - 7 Days"), (75, "3 - 5 Days"), (60, "1 - 3 Days")],
    "banana": [(90, "3 - 5 Days"), (75, "2 - 3 Days"), (60, "1 - 2 Days")],
    "apple":  [(90, "14 - 21 Days"), (75, "7 - 14 Days"), (60, "3 - 7 Days")],
    "lemon":  [(90, "14 - 21 Days"), (75, "7 - 14 Days"), (60, "3 - 7 Days")],
}
_SHELF_FALLBACK = "< 24 - 48 Hours (Clearance Recommended)"


def _shelf_life(cat, score):
    for threshold, text in _SHELF_LIFE[cat]:
        if score >= threshold:
            return text
    return _SHELF_FALLBACK


# ---------------------------------------------------------------------------
# 4. PUBLIC FUNCTION (same name / return keys as before, plus extras)
# ---------------------------------------------------------------------------

def get_detailed_freshness_report(image_bytes, category="Tomato", debug=False, **kwargs):
    cat = category.strip().lower()
    if cat not in _SCORERS:
        return {"error": f"Unsupported category '{category}'. Use: {list(_SCORERS)}"}

    try:
        img = _load_image(image_bytes)
        mask = _segment_fruit(img, cat)
        feats = _extract_features(img, mask, cat)
        score, stage, colour_match = _SCORERS[cat](feats)
        score = float(np.clip(score, 5, 98))

        # Blur: wrinkles/spots can't be verified, so never give a high score.
        blurry = feats["sharpness"] < BLUR_THRESHOLD
        if blurry:
            score = min(score, BLUR_SCORE_CAP)
            stage = f"Unverified - Blurry Image (best guess: {stage})"
        score = round(score, 1)

        # Confidence: does the fruit region actually look like the chosen category?
        area_ratio = float(mask.sum()) / mask.size
        confidence = "High"
        warning = None
        if blurry:
            confidence = "Low"
            warning = ("Image is blurry, so wrinkles and spots can't be checked reliably. "
                       "Score is capped; retake a sharp, well-lit photo.")
        elif colour_match < 0.35 or area_ratio < 0.10:
            confidence = "Low"
            warning = ("Fruit colour doesn't match the selected category or the fruit is "
                       "too small in the frame. Check the category / retake the photo.")

        avg = img[mask].mean(axis=0) if mask.any() else img.reshape(-1, 3).mean(axis=0)
        report = {
            "freshness_score_100": score,
            "classification": stage,
            "category": category,
            "shelf_life": _shelf_life(cat, score),
            "confidence": confidence,
            "image_sharp": not blurry,
            "metrics": {
                "avg_red": round(float(avg[0]), 2),
                "avg_green": round(float(avg[1]), 2),
                "avg_blue": round(float(avg[2]), 2),
                "brightness": round(feats["mean_v"], 2),
                "defect_percent": round(feats["defect"] * 100, 1),
                "gloss_percent": round(feats["gloss"] * 100, 1),
            },
        }
        if warning:
            report["warning"] = warning
        if debug:
            report["debug_features"] = {k: round(v, 3) for k, v in feats.items()}
            report["debug_wrinkle_severity"] = round(
                _wrinkle_severity(feats, APPLE_WRINKLE_CREASE, APPLE_WRINKLE_DETAIL)
                if cat == "apple" else _wrinkle_severity(feats), 2)
            report["debug_mask_area"] = round(area_ratio, 3)
        return report

    except Exception as e:
        # Do NOT invent a score on failure; a fake 80 hides real problems.
        return {"error": f"Could not analyse image: {e}", "category": category}