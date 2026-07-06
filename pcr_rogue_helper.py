"""
Automate the PCR rogue-like reroll flow in LDPlayer / Leidian 9.

The script reads the provided annotated PNG files in this directory:
1.png ... 9.png, detect3.png, detect5.png, boss31.png ... boss53.png.

It extracts red-square annotations as click/crop targets, recognizes the
current emulator screen with Pillow/Numpy, and clicks through ADB by default.
No OpenCV or PyAutoGUI dependency is required.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import html
import http.server
import io
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageGrab


if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
DEFAULT_COMBO_CONFIG = ROOT / "valid_combos.json"
DEFAULT_VALID_COMBOS = {
    ("boss51.png", "boss31.png"),
    ("boss51.png", "boss32.png"),
    ("boss51.png", "boss33.png"),
    ("boss53.png", "boss32.png"),
    ("boss53.png", "boss33.png"),
}
ACTIVE_VALID_COMBOS = set(DEFAULT_VALID_COMBOS)

DEFAULT_WINDOW_KEYWORDS = (
    "leidian",
    "ldplayer",
    "雷电",
    "雷電",
)

SCREEN_MATCH_THRESHOLD = 36.0
DETECT_MATCH_THRESHOLD = 42.0
BOSS_MATCH_THRESHOLD = 65.0
BOSS_MIN_CONFIDENCE = 60.0
BOSS_CONFIDENCE_SCORE_SCALE = 0.75
MIN_BOSS_GAP = 2.0
FUZZY_BLUR_RADIUS = 0.8
FUZZY_COMPARE_WIDTH = 640
TITLE_MATCH_THRESHOLD = 55.0
TITLE_GUARD_EXEMPT_SCREENS = {"5.png"}


user32 = ctypes.windll.user32
VK_SNAPSHOT = 0x2C
KEYEVENTF_KEYUP = 0x0002


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def inset(self, pixels: int) -> "Box":
        if self.width <= pixels * 2 or self.height <= pixels * 2:
            return self
        return Box(
            self.left + pixels,
            self.top + pixels,
            self.right - pixels,
            self.bottom - pixels,
        )

    def scaled(self, sx: float, sy: float) -> "Box":
        return Box(
            int(round(self.left * sx)),
            int(round(self.top * sy)),
            int(round(self.right * sx)),
            int(round(self.bottom * sy)),
        )


@dataclass
class AnnotatedImage:
    name: str
    path: Path
    image: Image.Image
    boxes: list[Box]
    screen_array: np.ndarray
    mask_array: np.ndarray
    fuzzy_array: np.ndarray
    fuzzy_mask_array: np.ndarray
    fuzzy_size: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass
class Match:
    name: str
    score: float
    title_score: float = 0.0


def red_mask(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    return (arr[:, :, 0] > 180) & (arr[:, :, 1] < 90) & (arr[:, :, 2] < 90)


def fuzzy_size_for(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width <= FUZZY_COMPARE_WIDTH:
        return size
    scaled_height = max(1, int(round(height * FUZZY_COMPARE_WIDTH / width)))
    return (FUZZY_COMPARE_WIDTH, scaled_height)


def fuzzy_image_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    fuzzy = image.convert("RGB").filter(ImageFilter.GaussianBlur(FUZZY_BLUR_RADIUS))
    fuzzy = fuzzy.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(fuzzy, dtype=np.float32)


def connected_red_boxes(image: Image.Image, min_pixels: int = 700) -> list[Box]:
    mask = red_mask(image)
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    boxes: list[Box] = []

    for y in range(height):
        xs = np.where(mask[y] & ~seen[y])[0]
        for start_x in xs:
            if seen[y, start_x] or not mask[y, start_x]:
                continue
            stack = [(int(start_x), y)]
            seen[y, start_x] = True
            min_x = max_x = int(start_x)
            min_y = max_y = y
            count = 0
            while stack:
                x, yy = stack.pop()
                count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, yy)
                max_y = max(max_y, yy)
                for nx in (x - 1, x, x + 1):
                    for ny in (yy - 1, yy, yy + 1):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and not seen[ny, nx]
                            and mask[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if count >= min_pixels:
                boxes.append(Box(min_x, min_y, max_x, max_y))

    return boxes


def sort_click_boxes(name: str, boxes: list[Box]) -> list[Box]:
    if name == "5.png" and len(boxes) > 1:
        # Requested special case: click multiple marks on screenshot 5 in
        # reading order. The top-left annotation currently spans the first two
        # character icons as one connected red component, so split that wide
        # component into two icon targets.
        main_boxes = sorted(boxes, key=lambda box: box.area, reverse=True)[:4]
        split_boxes: list[Box] = []
        for box in main_boxes:
            if box.top < 340 and box.width > 240:
                mid = (box.left + box.right) // 2
                split_boxes.append(Box(box.left, box.top, mid, box.bottom))
                split_boxes.append(Box(mid + 1, box.top, box.right, box.bottom))
            else:
                split_boxes.append(box)
        return sorted(split_boxes, key=lambda box: (box.top // 80, box.left))
    return sorted(boxes, key=lambda box: box.area, reverse=True)[:1]


def load_annotated(name: str) -> AnnotatedImage:
    path = ROOT / name
    image = Image.open(path).convert("RGB")
    boxes = sort_click_boxes(name, connected_red_boxes(image))
    if not boxes:
        raise RuntimeError(f"No red target box found in {name}")

    arr = np.asarray(image, dtype=np.float32)
    keep = ~red_mask(image)
    fuzzy_size = fuzzy_size_for(image.size)
    return AnnotatedImage(
        name=name,
        path=path,
        image=image,
        boxes=boxes,
        screen_array=arr,
        mask_array=keep,
        fuzzy_array=fuzzy_image_array(image, fuzzy_size),
        fuzzy_mask_array=resize_mask(keep, fuzzy_size),
        fuzzy_size=fuzzy_size,
    )


def resize_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.NEAREST)) > 0


def masked_mean_absdiff(live: Image.Image, ref: AnnotatedImage) -> float:
    live_ref_size = live.convert("RGB").resize(ref.size, Image.Resampling.BILINEAR)
    live_arr = fuzzy_image_array(live_ref_size, ref.fuzzy_size)
    diff = np.abs(live_arr - ref.fuzzy_array).mean(axis=2)
    usable = ref.fuzzy_mask_array
    if usable.sum() == 0:
        return float("inf")
    return float(diff[usable].mean())


def best_match(live: Image.Image, refs: Iterable[AnnotatedImage]) -> Match:
    best = Match("", float("inf"), float("inf"))
    for ref in refs:
        score = masked_mean_absdiff(live, ref)
        if score < best.score:
            best = Match(ref.name, score, title_anchor_score(live, ref))
    return best


def title_anchor_score(live: Image.Image, ref: AnnotatedImage) -> float:
    """Compare the top title/header area so similar modals do not alias."""
    live_arr = resize_array(live.convert("RGB"), ref.size)
    diff = np.abs(live_arr - ref.screen_array).mean(axis=2)
    width, height = ref.size
    left = int(width * 0.25)
    right = int(width * 0.75)
    top = int(height * 0.02)
    bottom = int(height * 0.16)
    region = np.zeros(ref.mask_array.shape, dtype=bool)
    region[top:bottom, left:right] = True
    usable = ref.mask_array & region
    if usable.sum() == 0:
        return float("inf")
    return float(diff[usable].mean())


def is_screen_match(live: Image.Image, ref: AnnotatedImage, threshold: float) -> tuple[bool, float, float]:
    score = masked_mean_absdiff(live, ref)
    title_score = title_anchor_score(live, ref)
    title_ok = ref.name in TITLE_GUARD_EXEMPT_SCREENS or title_score <= TITLE_MATCH_THRESHOLD
    return score <= threshold and title_ok, score, title_score


def plain_mean_absdiff(left: Image.Image, right: Image.Image) -> float:
    left_fuzzy = left.convert("RGB").filter(ImageFilter.GaussianBlur(1.2))
    right_fuzzy = right.convert("RGB").filter(ImageFilter.GaussianBlur(1.2))
    diff = ImageChops.difference(left_fuzzy, right_fuzzy)
    return float(np.asarray(diff, dtype=np.float32).mean())


def boss_icon_score(live_crop: Image.Image, template: Image.Image) -> float:
    """Find the best template-like patch inside a larger boss crop."""
    crop = live_crop.convert("RGB")
    best = float("inf")

    for scale in (0.85, 1.0, 1.15, 1.3):
        tw = max(8, int(round(template.width * scale)))
        th = max(8, int(round(template.height * scale)))
        if tw > crop.width or th > crop.height:
            continue
        tmpl = template.convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
        stride = 2 if max(tw, th) <= 95 else 3
        for y in range(0, crop.height - th + 1, stride):
            for x in range(0, crop.width - tw + 1, stride):
                patch = crop.crop((x, y, x + tw, y + th))
                score = plain_mean_absdiff(patch, tmpl)
                if score < best:
                    best = score

    if best < float("inf"):
        return best

    resized = crop.resize(template.size, Image.Resampling.BILINEAR)
    return plain_mean_absdiff(resized, template)


def find_window(title_part: str | None = None) -> tuple[int, str]:
    matches: list[tuple[int, str]] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        lowered = title.lower()
        if title_part:
            if title_part.lower() in lowered:
                matches.append((hwnd, title))
        elif any(keyword in lowered for keyword in DEFAULT_WINDOW_KEYWORDS):
            matches.append((hwnd, title))
        return True

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)(callback)
    user32.EnumWindows(enum_proc, 0)

    if not matches:
        wanted = title_part or " / ".join(DEFAULT_WINDOW_KEYWORDS)
        raise RuntimeError(f"Could not find a visible window matching: {wanted}")
    return matches[0]


def client_bbox(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")
    point = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise RuntimeError("ClientToScreen failed")
    return (
        point.x,
        point.y,
        point.x + rect.right - rect.left,
        point.y + rect.bottom - rect.top,
    )


def focus_window(hwnd: int) -> None:
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def grab_client(hwnd: int) -> Image.Image:
    return ImageGrab.grab(bbox=client_bbox(hwnd)).convert("RGB")


def press_print_screen() -> None:
    user32.keybd_event(VK_SNAPSHOT, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(VK_SNAPSHOT, 0, KEYEVENTF_KEYUP, 0)


def grab_client_via_keyboard(hwnd: int, timeout: float = 2.0) -> Image.Image:
    focus_window(hwnd)
    time.sleep(0.08)
    press_print_screen()

    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        grabbed = ImageGrab.grabclipboard()
        if isinstance(grabbed, Image.Image):
            left, top, right, bottom = client_bbox(hwnd)
            desktop = grabbed.convert("RGB")
            return desktop.crop((left, top, right, bottom))
        last_error = f"clipboard={type(grabbed).__name__}"
        time.sleep(0.08)
    raise RuntimeError(f"Print Screen did not produce a clipboard image ({last_error})")


def default_adb_path() -> str:
    candidates = [
        ROOT / "adb.exe",
        Path(r"D:\leidian\LDPlayer9\adb.exe"),
        Path(r"C:\leidian\LDPlayer9\adb.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("adb")
    if found:
        return found
    return "adb"


def adb_command(args: argparse.Namespace, *cmd: str, timeout: float = 10.0) -> subprocess.CompletedProcess[bytes]:
    base = [args.adb_path]
    if args.adb_serial:
        base.extend(["-s", args.adb_serial])
    return subprocess.run(
        base + list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def grab_adb(args: argparse.Namespace) -> Image.Image:
    proc = adb_command(args, "exec-out", "screencap", "-p", timeout=8.0)
    if proc.returncode == 0:
        data = proc.stdout.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass

    # Some older Android/LDPlayer adb builds mangle binary data through
    # exec-out. Pulling the file is slower but reliable.
    remote = "/sdcard/pcr_rogue_helper_screen.png"
    local = ROOT / ".pcr_rogue_helper_screen.png"
    cap = adb_command(args, "shell", "screencap", "-p", remote, timeout=8.0)
    if cap.returncode != 0:
        err = cap.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ADB screencap failed: {err}")
    pull = adb_command(args, "pull", remote, str(local), timeout=8.0)
    if pull.returncode != 0:
        err = pull.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ADB pull screenshot failed: {err}")
    return Image.open(local).convert("RGB")


def tap_adb(args: argparse.Namespace, x: int, y: int, dry_run: bool = False) -> None:
    print(f"adb tap=({x}, {y})")
    if dry_run:
        return
    proc = adb_command(args, "shell", "input", "tap", str(int(x)), str(int(y)), timeout=5.0)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ADB tap failed: {err}")


def click_screen(x: int, y: int, dry_run: bool = False) -> None:
    print(f"click screen=({x}, {y})")
    if dry_run:
        return
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.04)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


def click_box_window(hwnd: int, live: Image.Image, ref: AnnotatedImage, box: Box, dry_run: bool) -> None:
    bbox = client_bbox(hwnd)
    live_w, live_h = live.size
    ref_w, ref_h = ref.size
    sx = live_w / ref_w
    sy = live_h / ref_h
    cx, cy = box.center
    click_screen(
        bbox[0] + int(round(cx * sx)),
        bbox[1] + int(round(cy * sy)),
        dry_run=dry_run,
    )


def click_box_adb(args: argparse.Namespace, live: Image.Image, ref: AnnotatedImage, box: Box) -> None:
    live_w, live_h = live.size
    ref_w, ref_h = ref.size
    sx = live_w / ref_w
    sy = live_h / ref_h
    cx, cy = box.center
    tap_adb(args, int(round(cx * sx)), int(round(cy * sy)), dry_run=args.dry_run)


def skip_unknown_window(hwnd: int, dry_run: bool = False) -> None:
    left, top, right, bottom = client_bbox(hwnd)
    width = right - left
    height = bottom - top
    click_screen(left + int(width * 0.16), top + int(height * 0.50), dry_run=dry_run)


def skip_unknown_adb(args: argparse.Namespace, live: Image.Image) -> None:
    width, height = live.size
    tap_adb(args, int(width * 0.16), int(height * 0.50), dry_run=args.dry_run)


def skip_left(args: argparse.Namespace, hwnd: int | None, live: Image.Image) -> None:
    if args.click_mode == "adb":
        skip_unknown_adb(args, live)
    else:
        skip_unknown_window(hwnd, args.dry_run)  # type: ignore[arg-type]


def crop_scaled(live: Image.Image, ref: AnnotatedImage, box: Box) -> Image.Image:
    live_w, live_h = live.size
    ref_w, ref_h = ref.size
    sx = live_w / ref_w
    sy = live_h / ref_h
    scaled = box.inset(6).scaled(sx, sy)
    return live.crop((scaled.left, scaled.top, scaled.right + 1, scaled.bottom + 1))


def classify_boss(
    live: Image.Image,
    detect_ref: AnnotatedImage,
    boss_templates: dict[str, Image.Image],
) -> tuple[str | None, list[tuple[str, float]]]:
    crop = crop_scaled(live, detect_ref, detect_ref.boxes[0])
    scores = sorted(
        ((name, boss_icon_score(crop, image)) for name, image in boss_templates.items()),
        key=lambda item: item[1],
    )
    if not scores:
        return None, []
    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else float("inf")
    if boss_match_confidence(best_score) < BOSS_MIN_CONFIDENCE:
        return None, scores
    if best_score <= BOSS_MATCH_THRESHOLD and second_score - best_score >= MIN_BOSS_GAP:
        return best_name, scores
    return None, scores


def boss_match_confidence(score: float) -> float:
    return max(0.0, min(100.0, 100.0 - score * BOSS_CONFIDENCE_SCORE_SCALE))


def format_boss_scores(scores: list[tuple[str, float]]) -> str:
    return str(
        [
            (name, round(score, 2), round(boss_match_confidence(score), 1))
            for name, score in scores[:3]
        ]
    )


def valid_combo(boss5: str | None, boss3: str | None) -> bool:
    return boss5 is not None and boss3 is not None and (boss5, boss3) in ACTIVE_VALID_COMBOS


def load_boss_templates(prefix: str) -> dict[str, Image.Image]:
    return {
        path.name: Image.open(path).convert("RGB")
        for path in sorted(ROOT.glob(f"{prefix}*.png"))
    }


def sorted_boss_names(templates: dict[str, Image.Image]) -> list[str]:
    return sorted(templates, key=lambda name: (len(name), name))


def boss_display_name(filename: str) -> str:
    stem = filename.replace(".png", "")
    if stem.startswith("boss3"):
        return f"三王{stem.removeprefix('boss3')}"
    if stem.startswith("boss5"):
        return f"五王{stem.removeprefix('boss5')}"
    return "未知"


def load_combo_config(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("valid_combinations", [])
    combos: set[tuple[str, str]] = set()
    for pair in pairs:
        if isinstance(pair, dict):
            boss5 = pair.get("boss5")
            boss3 = pair.get("boss3")
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            boss5, boss3 = pair
        else:
            continue
        if isinstance(boss5, str) and isinstance(boss3, str):
            combos.add((boss5, boss3))
    return combos


def save_combo_config(path: Path, combos: set[tuple[str, str]]) -> None:
    payload = {
        "valid_combinations": [
            {"boss5": boss5, "boss3": boss3}
            for boss5, boss3 in sorted(combos)
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_or_configured_combos(path_text: str | None) -> set[tuple[str, str]]:
    if not path_text:
        return set(DEFAULT_VALID_COMBOS)
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise RuntimeError(f"Combo config not found: {path}")
    combos = load_combo_config(path)
    if not combos:
        raise RuntimeError(f"Combo config has no valid combinations: {path}")
    return combos


def configure_combos_ui(
    boss3_templates: dict[str, Image.Image],
    boss5_templates: dict[str, Image.Image],
    output_path: Path,
    initial_combos: set[tuple[str, str]],
) -> None:
    boss3_names = sorted_boss_names(boss3_templates)
    boss5_names = sorted_boss_names(boss5_templates)

    def icon_data_uri(image: Image.Image) -> str:
        preview = image.resize((64, 64), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    boss3_icons = {name: icon_data_uri(boss3_templates[name]) for name in boss3_names}
    boss5_icons = {name: icon_data_uri(boss5_templates[name]) for name in boss5_names}
    done = threading.Event()
    result: dict[str, str] = {}

    def checkbox_name(boss5: str, boss3: str) -> str:
        return f"{boss5}|{boss3}"

    def render_page(saved: bool = False) -> bytes:
        header_cells = []
        for boss3 in boss3_names:
            header_cells.append(
                f"""
                <th>
                  <img src="{boss3_icons[boss3]}" alt="{html.escape(boss_display_name(boss3))}">
                  <div>{html.escape(boss_display_name(boss3))}</div>
                </th>
                """
            )
        rows = []
        for boss5 in boss5_names:
            cells = []
            for boss3 in boss3_names:
                name = checkbox_name(boss5, boss3)
                checked = " checked" if (boss5, boss3) in initial_combos else ""
                cells.append(
                    f'<td><label><input type="checkbox" name="{html.escape(name)}"{checked}> 有效</label></td>'
                )
            rows.append(
                f"""
                <tr>
                  <th>
                    <img src="{boss5_icons[boss5]}" alt="{html.escape(boss_display_name(boss5))}">
                    <div>{html.escape(boss_display_name(boss5))}</div>
                  </th>
                  {''.join(cells)}
                </tr>
                """
            )
        notice = "<p class=\"notice\">已保存，可以关闭这个页面。</p>" if saved else ""
        page = f"""
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <title>有效组合设置</title>
          <style>
            body {{ font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2937; }}
            h1 {{ font-size: 22px; margin: 0 0 12px; }}
            p {{ margin: 0 0 16px; color: #4b5563; }}
            table {{ border-collapse: collapse; }}
            th, td {{ border: 1px solid #d1d5db; padding: 10px 12px; text-align: center; min-width: 96px; }}
            th {{ background: #f9fafb; font-weight: 600; }}
            img {{ width: 64px; height: 64px; object-fit: contain; display: block; margin: 0 auto 6px; }}
            label {{ cursor: pointer; white-space: nowrap; }}
            .actions {{ display: flex; gap: 10px; margin-top: 18px; }}
            button {{ font-size: 14px; padding: 8px 14px; cursor: pointer; }}
            .notice {{ color: #047857; font-weight: 600; }}
          </style>
        </head>
        <body>
          <h1>请选择有效组合</h1>
          <p>勾选后点击“保存并退出”。保存位置：{html.escape(str(output_path))}</p>
          {notice}
          <form method="post" action="/save">
            <table>
              <tr><th>五王 / 三王</th>{''.join(header_cells)}</tr>
              {''.join(rows)}
            </table>
            <div class="actions">
              <button type="submit">保存并退出</button>
              <button type="submit" formaction="/defaults">恢复默认</button>
              <button type="submit" formaction="/cancel">取消</button>
            </div>
          </form>
        </body>
        </html>
        """
        return page.encode("utf-8")

    class ComboHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.send_html(render_page())

        def do_POST(self) -> None:
            nonlocal initial_combos
            length = int(self.headers.get("Content-Length", "0"))
            fields = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
            if self.path == "/cancel":
                result["status"] = "cancelled"
                done.set()
                self.send_html(render_page())
                return
            if self.path == "/defaults":
                initial_combos = set(DEFAULT_VALID_COMBOS)
                self.send_html(render_page())
                return
            combos = {
                (boss5, boss3)
                for boss5 in boss5_names
                for boss3 in boss3_names
                if checkbox_name(boss5, boss3) in fields
            }
            save_combo_config(output_path, combos)
            result["status"] = "saved"
            done.set()
            self.send_html(render_page(saved=True))

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), ComboHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"配置界面已打开：{url}")
        webbrowser.open(url)
        done.wait()
        server.shutdown()
        thread.join(timeout=2.0)

    if result.get("status") == "saved":
        print(f"已保存到：{output_path}")
    else:
        print("已取消，未保存配置。")


def print_targets(sequence: list[AnnotatedImage], detect3: AnnotatedImage, detect5: AnnotatedImage) -> None:
    print("Loaded target boxes:")
    for ref in sequence:
        centers = [tuple(round(v, 1) for v in box.center) for box in ref.boxes]
        print(f"  {ref.name}: size={ref.size}, boxes={ref.boxes}, centers={centers}")
    print(f"  detect3.png crop: {detect3.boxes[0]}")
    print(f"  detect5.png crop: {detect5.boxes[0]}")


def run(args: argparse.Namespace) -> int:
    global ACTIVE_VALID_COMBOS
    global SCREEN_MATCH_THRESHOLD, DETECT_MATCH_THRESHOLD, BOSS_MATCH_THRESHOLD, BOSS_MIN_CONFIDENCE, MIN_BOSS_GAP
    global FUZZY_BLUR_RADIUS, FUZZY_COMPARE_WIDTH, TITLE_MATCH_THRESHOLD
    SCREEN_MATCH_THRESHOLD = args.screen_match_threshold
    DETECT_MATCH_THRESHOLD = args.detect_match_threshold
    BOSS_MATCH_THRESHOLD = args.boss_match_threshold
    BOSS_MIN_CONFIDENCE = args.boss_min_confidence
    MIN_BOSS_GAP = args.min_boss_gap
    FUZZY_BLUR_RADIUS = args.fuzzy_blur_radius
    FUZZY_COMPARE_WIDTH = args.fuzzy_compare_width
    TITLE_MATCH_THRESHOLD = args.title_match_threshold

    sequence = [load_annotated(f"{i}.png") for i in range(1, 10)]
    sequence_index = {ref.name: index for index, ref in enumerate(sequence)}
    detect3 = load_annotated("detect3.png")
    detect5 = load_annotated("detect5.png")
    boss3_templates = load_boss_templates("boss3")
    boss5_templates = load_boss_templates("boss5")

    if not boss3_templates or not boss5_templates:
        raise RuntimeError("Missing boss template images")

    combo_path = Path(args.combo_config) if args.combo_config else DEFAULT_COMBO_CONFIG
    if not combo_path.is_absolute():
        combo_path = ROOT / combo_path

    initial_combos = set(DEFAULT_VALID_COMBOS)
    if args.combo_config and combo_path.exists():
        initial_combos = load_combo_config(combo_path)

    if args.configure_combos:
        configure_combos_ui(boss3_templates, boss5_templates, combo_path, initial_combos)
        return 0

    ACTIVE_VALID_COMBOS = default_or_configured_combos(args.combo_config)
    print(f"Valid combos: {sorted(ACTIVE_VALID_COMBOS)}")

    if args.print_targets:
        print_targets(sequence, detect3, detect5)

    hwnd: int | None = None
    if args.capture in {"window", "keyboard"} or args.click_mode == "window":
        hwnd, title = find_window(args.window_title)
        print(f"Using window: hwnd={hwnd}, title={title!r}")
        focus_window(hwnd)

    if args.capture == "adb" or args.click_mode == "adb":
        args.adb_path = args.adb_path or default_adb_path()
        print(f"Using ADB: {args.adb_path} serial={args.adb_serial or 'default'}")
        probe = adb_command(args, "shell", "wm", "size", timeout=5.0)
        if probe.returncode == 0:
            print(probe.stdout.decode("utf-8", errors="replace").strip())
        else:
            err = probe.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ADB is not ready: {err}")
    time.sleep(args.start_delay)

    next_index = 0
    boss5: str | None = None
    boss3: str | None = None
    waiting_detect5 = False
    waiting_detect3 = False
    iteration = 1
    step = 0
    last_skip = 0.0

    while True:
        step += 1
        if args.max_steps and step > args.max_steps:
            print(f"Reached --max-steps={args.max_steps}; exiting.")
            return 0
        if args.max_loops and iteration > args.max_loops:
            print(f"Reached --max-loops={args.max_loops}; exiting.")
            return 0

        if args.capture == "adb":
            live = grab_adb(args)
        elif args.capture == "keyboard":
            live = grab_client_via_keyboard(hwnd)  # type: ignore[arg-type]
        else:
            live = grab_client(hwnd)  # type: ignore[arg-type]

        if waiting_detect5:
            score = masked_mean_absdiff(live, detect5)
            print(f"iteration={iteration} waiting detect5 score={score:.2f}")
            if score <= DETECT_MATCH_THRESHOLD:
                boss5, scores = classify_boss(live, detect5, boss5_templates)
                print(f"boss5={boss5 or 'invalid'} scores/conf={format_boss_scores(scores)}")
                waiting_detect5 = False
                print("Closing detect5 modal and returning to the map.")
                skip_left(args, hwnd, live)
                if boss5 not in {"boss51.png", "boss53.png"}:
                    print("boss5 cannot make a valid combo; continuing at screen 8.")
                    next_index = sequence_index["8.png"]
                    time.sleep(args.after_click_delay)
                    continue
                next_index = sequence_index["7.png"]
                time.sleep(args.after_click_delay)
                continue

        if waiting_detect3:
            score = masked_mean_absdiff(live, detect3)
            print(f"iteration={iteration} waiting detect3 score={score:.2f}")
            if score <= DETECT_MATCH_THRESHOLD:
                boss3, scores = classify_boss(live, detect3, boss3_templates)
                print(f"boss3={boss3 or 'invalid'} scores/conf={format_boss_scores(scores)}")
                waiting_detect3 = False
                print("Closing detect3 modal and returning to screen 8.")
                skip_left(args, hwnd, live)
                if valid_combo(boss5, boss3):
                    print(
                        f"VALID COMBO: boss5={boss5}, boss3={boss3}. "
                        "Confirmed target start; exiting script after closing detect3 modal."
                    )
                    return 0
                print("Invalid combo; continuing at screen 8.")
                next_index = sequence_index["8.png"]
                time.sleep(args.after_click_delay)
                continue

        expected = sequence[next_index]
        expected_ok, expected_score, expected_title_score = is_screen_match(
            live, expected, SCREEN_MATCH_THRESHOLD
        )
        all_match = best_match(live, sequence + [detect3, detect5])
        print(
            f"iteration={iteration} expect={expected.name} "
            f"expect_score={expected_score:.2f} title={expected_title_score:.2f} "
            f"best={all_match.name}:{all_match.score:.2f}/title={all_match.title_score:.2f}"
        )

        if (
            not expected_ok
            and all_match.name in sequence_index
            and all_match.score <= SCREEN_MATCH_THRESHOLD
            and (
                all_match.name in TITLE_GUARD_EXEMPT_SCREENS
                or all_match.title_score <= TITLE_MATCH_THRESHOLD
            )
        ):
            next_index = sequence_index[all_match.name]
            print(f"Resynced to recognized screen {all_match.name}.")
            time.sleep(args.poll_interval)
            continue

        if not expected_ok and all_match.name == "detect5.png" and all_match.score <= DETECT_MATCH_THRESHOLD:
            boss5, scores = classify_boss(live, detect5, boss5_templates)
            print(f"Resynced to detect5.png. boss5={boss5 or 'invalid'} scores/conf={format_boss_scores(scores)}")
            print("Closing detect5 modal and returning to the map.")
            skip_left(args, hwnd, live)
            if boss5 not in {"boss51.png", "boss53.png"}:
                print("boss5 cannot make a valid combo; continuing at screen 8.")
                next_index = sequence_index["8.png"]
            else:
                next_index = sequence_index["7.png"]
            time.sleep(args.after_click_delay)
            continue

        if not expected_ok and all_match.name == "detect3.png" and all_match.score <= DETECT_MATCH_THRESHOLD:
            boss3, scores = classify_boss(live, detect3, boss3_templates)
            print(f"Resynced to detect3.png. boss3={boss3 or 'invalid'} scores/conf={format_boss_scores(scores)}")
            print("Closing detect3 modal and returning to screen 8.")
            skip_left(args, hwnd, live)
            if valid_combo(boss5, boss3):
                print(
                    f"VALID COMBO: boss5={boss5}, boss3={boss3}. "
                    "Confirmed target start; exiting script after closing detect3 modal."
                )
                return 0
            print("Invalid combo; continuing at screen 8.")
            next_index = sequence_index["8.png"]
            time.sleep(args.after_click_delay)
            continue

        if expected_ok:
            if expected.name == "8.png" and valid_combo(boss5, boss3):
                print(
                    f"VALID COMBO: boss5={boss5}, boss3={boss3}. "
                    "At screenshot 8; exiting script without clicking its button."
                )
                return 0

            for box in expected.boxes:
                if args.click_mode == "adb":
                    click_box_adb(args, live, expected, box)
                else:
                    click_box_window(hwnd, live, expected, box, args.dry_run)  # type: ignore[arg-type]
                time.sleep(args.multi_click_delay)

            if expected.name == "6.png":
                waiting_detect5 = True
            elif expected.name == "7.png":
                waiting_detect3 = True
            elif expected.name == "9.png":
                iteration += 1
                boss5 = None
                boss3 = None
                next_index = 0
                time.sleep(args.after_click_delay)
                continue

            next_index = (next_index + 1) % len(sequence)
            if next_index == 0:
                iteration += 1
            time.sleep(args.after_click_delay)
            continue

        now = time.time()
        if now - last_skip >= args.unknown_skip_interval:
            print("Unknown/interstitial screen; sending left-side skip click.")
            skip_left(args, hwnd, live)
            last_skip = now

        time.sleep(args.poll_interval)
        if next_index == 0 and not waiting_detect3 and not waiting_detect5:
            iteration += 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCR rogue helper for LDPlayer / Leidian 9."
    )
    parser.add_argument(
        "--window-title",
        help="Part of the LDPlayer window title. Defaults to Leidian/LDPlayer keywords.",
    )
    parser.add_argument(
        "--capture",
        choices=("keyboard", "window", "adb"),
        default="adb",
        help="Use ADB screenshots, desktop window capture, or Print Screen keyboard capture.",
    )
    parser.add_argument(
        "--click-mode",
        choices=("adb", "window"),
        default="adb",
        help="Use ADB input taps or Windows desktop mouse clicks.",
    )
    parser.add_argument(
        "--adb-path",
        default=None,
        help=r"Path to adb.exe. Defaults to D:\leidian\LDPlayer9\adb.exe when present.",
    )
    parser.add_argument(
        "--adb-serial",
        default="emulator-5554",
        help="ADB device serial. Use an empty string to let adb choose the default device.",
    )
    parser.add_argument(
        "--configure-combos",
        action="store_true",
        help="Open the Chinese combo configuration UI, save the config, then exit.",
    )
    parser.add_argument(
        "--combo-config",
        default=None,
        help="Path to a saved valid combo JSON config. Omit to use the built-in default rules.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log clicks without sending them.")
    parser.add_argument("--print-targets", action="store_true", help="Print extracted red boxes.")
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=0.7)
    parser.add_argument("--after-click-delay", type=float, default=1.2)
    parser.add_argument("--multi-click-delay", type=float, default=0.35)
    parser.add_argument("--unknown-skip-interval", type=float, default=2.0)
    parser.add_argument("--screen-match-threshold", type=float, default=SCREEN_MATCH_THRESHOLD)
    parser.add_argument("--detect-match-threshold", type=float, default=DETECT_MATCH_THRESHOLD)
    parser.add_argument("--boss-match-threshold", type=float, default=BOSS_MATCH_THRESHOLD)
    parser.add_argument("--boss-min-confidence", type=float, default=BOSS_MIN_CONFIDENCE)
    parser.add_argument("--min-boss-gap", type=float, default=MIN_BOSS_GAP)
    parser.add_argument("--fuzzy-blur-radius", type=float, default=FUZZY_BLUR_RADIUS)
    parser.add_argument("--fuzzy-compare-width", type=int, default=FUZZY_COMPARE_WIDTH)
    parser.add_argument("--title-match-threshold", type=float, default=TITLE_MATCH_THRESHOLD)
    parser.add_argument(
        "--max-loops",
        type=int,
        default=0,
        help="Stop after this many outer loop passes. 0 means run forever.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many screen checks. Useful with --dry-run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
