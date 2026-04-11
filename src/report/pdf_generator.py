"""
DeepCon-M15X PDF Report Generator
==================================
Streamlit 리포트 탭에서 호출하여 PDF 다운로드를 생성한다.
fpdf2 + matplotlib 기반. 한글 지원 (NotoSansKR + AppleSDGothicNeo Bold).

Sections:
1. Cover page
2. Executive Summary (AI-generated)
3. KPI summary + glossary
4. Transit time analysis (MAT/LBT/EOD) + AI commentary
5. Equipment OPR (BP-level) + AI commentary
6. Worker analysis (EWI/CRE) + AI commentary
7. Space congestion + AI commentary
8. Conclusion / Notes
"""
from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

# ── Brand colours (RGB 0-255) ──
NAVY   = (26, 58, 92)
ACCENT = (0, 174, 239)
WHITE  = (255, 255, 255)
SLATE  = (100, 116, 139)
LBG    = (240, 244, 250)
BDR    = (200, 210, 225)
GREEN  = (0, 200, 151)
RED    = (255, 76, 76)
AMBER  = (255, 179, 0)
DARK_BG = (13, 27, 42)
LIGHT_ACCENT = (230, 245, 255)
SEPARATOR = (180, 195, 215)

# ── Page geometry (A4, mm) ──
MARG = 14
PBT  = 297 - MARG
CONTENT_W = 210 - 2 * MARG  # 182mm

# ── Font discovery ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FONT_SEARCH_PATHS = [
    os.path.join(_PROJECT_ROOT, "assets", "fonts", "NotoSansKR.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    os.path.expanduser("~/Library/Fonts/NanumGothic.otf"),
    os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"),
]

_FONT_BOLD_SEARCH_PATHS = [
    os.path.join(_PROJECT_ROOT, "assets", "fonts", "AppleSDGothicNeo.ttc"),
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_APPLE_BOLD_FONT_INDEX = 6

_NANUM_PATH = None
for p in _FONT_SEARCH_PATHS:
    if os.path.exists(p):
        _NANUM_PATH = p
        break

_NANUM_BOLD_PATH = None
_NANUM_BOLD_INDEX = None
for p in _FONT_BOLD_SEARCH_PATHS:
    if os.path.exists(p):
        _NANUM_BOLD_PATH = p
        _NANUM_BOLD_INDEX = _APPLE_BOLD_FONT_INDEX
        break
if _NANUM_BOLD_PATH is None:
    _NANUM_BOLD_PATH = _NANUM_PATH
    _NANUM_BOLD_INDEX = None

_HAS_NANUM = _NANUM_PATH is not None


def _m(t: tuple) -> tuple:
    """RGB 0-255 to matplotlib 0-1."""
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)


def _cl(s: str) -> str:
    """Clean string for PDF output."""
    if not s:
        return ""
    s = str(s)
    replacements = {
        "\u2014": "--", "\u2013": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u2192": "->",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    if not _HAS_NANUM:
        s = "".join(c if ord(c) < 256 else "?" for c in s)
    return s


def _fig_to_bytes(fig) -> bytes:
    """Render matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _tmp_png(data: bytes) -> str:
    """Write bytes to a temp PNG file, return path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path


# =====================================================================
# DeepConPDF class
# =====================================================================
class DeepConPDF(FPDF):
    """FPDF subclass with Korean font support, adaptive layout, and rich helpers."""

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_margins(MARG, MARG, MARG)
        self.set_auto_page_break(True, MARG)

        if _HAS_NANUM:
            self.add_font("Nanum", "", _NANUM_PATH)
            if _NANUM_BOLD_PATH and _NANUM_BOLD_INDEX is not None:
                self.add_font("Nanum", "B", _NANUM_BOLD_PATH, collection_font_number=_NANUM_BOLD_INDEX)
            elif _NANUM_BOLD_PATH:
                self.add_font("Nanum", "B", _NANUM_BOLD_PATH)
            else:
                self.add_font("Nanum", "B", _NANUM_PATH)
            self._fn = "Nanum"
        else:
            self._fn = "Helvetica"

        self._period = ""
        self._title = "Deep Con at M15X"

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 8, "F")
        self.set_y(1.5)
        self.set_text_color(*WHITE)
        self.set_font(self._fn, "B", 7)
        self.cell(0, 5, f"  {self._title}  |  {self._period}", ln=True)
        self.set_text_color(0, 0, 0)
        self.set_y(12)

    def footer(self):
        self.set_y(-10)
        self.set_font(self._fn, "", 6.5)
        self.set_text_color(*SLATE)
        self.cell(0, 4,
                  f"TJLABS Corp.  |  Page {self.page_no()}/{{nb}}",
                  align="C")
        self.set_text_color(0, 0, 0)

    # ── Adaptive space check ──
    def _need_page(self, needed_h: float) -> None:
        """Add a new page if remaining space is less than needed_h mm."""
        if self.get_y() + needed_h > PBT:
            self.add_page()

    # ── Separator line ──
    def separator(self):
        """Draw a subtle horizontal separator line."""
        y = self.get_y() + 2
        self.set_draw_color(*SEPARATOR)
        self.set_line_width(0.2)
        self.line(MARG, y, MARG + CONTENT_W, y)
        self.set_y(y + 3)
        self.set_draw_color(0, 0, 0)

    # ── Cover page ──
    def add_cover(self, title: str, period: str, generated_at: str):
        self._period = period
        self.add_page()
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 297, "F")

        # Title
        self.set_y(80)
        self.set_text_color(*ACCENT)
        self.set_font(self._fn, "B", 28)
        self.cell(0, 14, _cl(title), align="C", ln=True)

        # Subtitle
        self.set_text_color(*WHITE)
        self.set_font(self._fn, "", 14)
        self.cell(0, 10, _cl("Spatial Data Analysis Report"), align="C", ln=True)

        # Period
        self.ln(10)
        self.set_text_color(175, 190, 220)
        self.set_font(self._fn, "", 11)
        self.cell(0, 8, _cl(period), align="C", ln=True)

        # Generated
        self.ln(5)
        self.set_text_color(120, 140, 170)
        self.set_font(self._fn, "", 9)
        self.cell(0, 6, _cl(f"Generated: {generated_at}"), align="C", ln=True)

        # Footer on cover
        self.set_y(255)
        self.set_text_color(100, 120, 150)
        self.set_font(self._fn, "", 8)
        self.cell(0, 5, _cl("TJLABS Corp.  |  jeffery@tjlabscorp.com"), align="C", ln=True)
        self.ln(2)
        self.set_text_color(80, 100, 130)
        self.set_font(self._fn, "", 7)
        self.cell(0, 4, _cl("Confidential - For internal use only"), align="C", ln=True)
        self.set_text_color(0, 0, 0)

    # ── Section header (numbered) ──
    def section_title(self, num: int, title: str):
        self._need_page(60)  # 제목+콘텐츠가 같은 페이지에 오도록 여유 확보
        self.ln(5)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font(self._fn, "B", 11)
        self.cell(0, 8, f"  {num}. {_cl(title)}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    # ── Sub-section header ──
    def sub_title(self, title: str):
        self._need_page(30)  # 소제목+콘텐츠가 같은 페이지에 오도록 여유 확보
        self.ln(2)
        self.set_font(self._fn, "B", 9)
        self.set_text_color(*NAVY)
        self.cell(0, 5, _cl(title), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    # ── KPI card row ──
    def kpi_row(self, cards: list[tuple[str, str, tuple]]):
        """Render a row of KPI cards. Each card = (label, value, color)."""
        n = len(cards)
        if n == 0:
            return
        cw = CONTENT_W / n
        self._need_page(24)
        y = self.get_y()

        for i, (label, value, color) in enumerate(cards):
            x = MARG + i * cw
            self.set_fill_color(*LBG)
            self.rect(x, y, cw - 2, 18, "F")
            self.set_fill_color(*color)
            self.rect(x, y, cw - 2, 1.5, "F")
            self.set_xy(x + 2, y + 2.5)
            self.set_font(self._fn, "", 7)
            self.set_text_color(*SLATE)
            self.cell(cw - 4, 4, _cl(label))
            self.set_xy(x + 2, y + 8)
            self.set_font(self._fn, "B", 13)
            self.set_text_color(*NAVY)
            self.cell(cw - 4, 7, _cl(value))

        self.set_text_color(0, 0, 0)
        self.set_y(y + 22)

    # ── Text paragraph ──
    def text_block(self, text: str, indent: float = 0, size: float = 8.5):
        self.set_x(MARG + indent)
        self.set_font(self._fn, "", size)
        self.set_text_color(*SLATE)
        self.multi_cell(CONTENT_W - indent, 4.8, _cl(text))
        self.set_text_color(0, 0, 0)

    # ── Parameter description (small muted text above/below chart) ──
    def param_desc(self, text: str):
        """Render a metric/parameter description as a small muted text block."""
        self._need_page(10)
        self.set_fill_color(*LIGHT_ACCENT)
        y0 = self.get_y()
        # estimate height
        chars_per_line = 100
        lines = text.split("\n")
        n_lines = sum(max(1, (len(line) // chars_per_line) + 1) for line in lines)
        h = n_lines * 4.0 + 6
        self.rect(MARG, y0, CONTENT_W, h, "F")
        self.set_xy(MARG + 3, y0 + 2)
        self.set_font(self._fn, "", 7)
        self.set_text_color(60, 80, 110)
        self.multi_cell(CONTENT_W - 6, 4.0, _cl(text))
        self.set_text_color(0, 0, 0)
        self.set_y(max(self.get_y() + 1, y0 + h + 1))

    # ── Chart image (adaptive) ──
    def add_chart(self, png_bytes: bytes, w: float = 0, h: float = 0):
        """Insert a chart image from PNG bytes.
        w=0: use CONTENT_W. h=0: auto aspect ratio from PNG dimensions.
        Includes intelligent page-break: chart will not be split across pages.
        """
        from PIL import Image as _PilImage
        import io as _io
        if w <= 0:
            w = CONTENT_W
        path = _tmp_png(png_bytes)
        try:
            with _PilImage.open(_io.BytesIO(png_bytes)) as im:
                px_w, px_h = im.size
            h_est = w * px_h / px_w if h == 0 else h
            self._need_page(h_est + 8)
            self.image(path, x=MARG, w=w, h=h)
            self.ln(3)
        finally:
            os.unlink(path)

    # ── Chart + commentary block (atomic: won't split across pages) ──
    def add_chart_with_commentary(
        self,
        png_bytes: bytes,
        commentary: str = "",
        param_description: str = "",
        w: float = 0,
    ):
        """Insert param description + chart + AI commentary as one atomic block.
        If the whole block doesn't fit, start a new page first.
        """
        from PIL import Image as _PilImage
        import io as _io
        if w <= 0:
            w = CONTENT_W

        # Estimate total height
        with _PilImage.open(_io.BytesIO(png_bytes)) as im:
            px_w, px_h = im.size
        chart_h = w * px_h / px_w
        param_h = self._estimate_text_h(param_description, 100, 4.0, 6) if param_description else 0
        commentary_h = self._estimate_text_h(commentary, 85, 5.0, 14) if commentary else 0
        total_h = param_h + chart_h + commentary_h + 12

        # If total exceeds page, at least start from fresh page
        if total_h > PBT - 20:
            # Block is very large, render sequentially with page breaks
            if param_description:
                self.param_desc(param_description)
            self.add_chart(png_bytes, w=w)
            if commentary:
                self.ai_box(commentary)
        else:
            self._need_page(total_h)
            if param_description:
                self.param_desc(param_description)
            path = _tmp_png(png_bytes)
            try:
                self.image(path, x=MARG, w=w, h=0)
                self.ln(3)
            finally:
                os.unlink(path)
            if commentary:
                self.ai_box(commentary)

    def _estimate_text_h(self, text: str, chars_per_line: int, line_h: float, overhead: float) -> float:
        """Estimate rendered height of a text block in mm."""
        if not text:
            return 0
        raw_lines = text.split("\n")
        total_lines = sum(max(1, (len(line) // chars_per_line) + 1) for line in raw_lines)
        return max(overhead + 4, total_lines * line_h + overhead)

    # ── Simple table ──
    def simple_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] = None):
        """Render a simple table with headers and rows."""
        n = len(headers)
        if col_widths is None:
            col_widths = [CONTENT_W / n] * n

        self._need_page(14)
        # Header
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font(self._fn, "B", 7.5)
        y = self.get_y()
        for i, h in enumerate(headers):
            x = MARG + sum(col_widths[:i])
            self.rect(x, y, col_widths[i], 6, "F")
            self.set_xy(x + 1, y + 1)
            self.cell(col_widths[i] - 2, 4, _cl(h), align="C")
        self.set_y(y + 6)

        # Rows
        self.set_text_color(40, 40, 40)
        for ri, row in enumerate(rows):
            if self.get_y() + 6 > PBT:
                self.add_page()
            y = self.get_y()
            bg = LBG if ri % 2 == 0 else WHITE
            self.set_fill_color(*bg)
            self.set_font(self._fn, "", 7.5)
            for ci, cell in enumerate(row):
                x = MARG + sum(col_widths[:ci])
                self.rect(x, y, col_widths[ci], 5.5, "F")
                self.set_xy(x + 1, y + 0.8)
                self.cell(col_widths[ci] - 2, 4, _cl(str(cell)), align="C")
            self.set_y(y + 5.5)

        self.set_text_color(0, 0, 0)
        self.ln(3)

    # ── AI commentary box ──
    def ai_box(self, text: str):
        """Render an AI analysis box with accent styling."""
        if not text:
            return
        self.ln(2)

        chars_per_line = 85
        raw_lines = text.split("\n")
        total_lines = sum(max(1, (len(line) // chars_per_line) + 1) for line in raw_lines)
        h = max(20, total_lines * 4.8 + 14)

        self._need_page(h + 4)
        y = self.get_y()

        # Background
        self.set_fill_color(240, 248, 255)
        self.rect(MARG, y, CONTENT_W, h, "F")
        # Accent bar (left)
        self.set_fill_color(*ACCENT)
        self.rect(MARG, y, 2.5, h, "F")
        # Label chip
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font(self._fn, "B", 6.5)
        self.set_xy(MARG + 4, y + 2)
        self.cell(20, 3.5, "  AI Analysis", fill=True)
        # Content
        self.set_xy(MARG + 5, y + 9)
        self.set_font(self._fn, "", 8)
        self.set_text_color(50, 65, 85)
        self.multi_cell(CONTENT_W - 10, 4.8, _cl(text))
        self.set_text_color(0, 0, 0)
        self.set_y(max(self.get_y() + 3, y + h + 3))

    # ── Executive summary box ──
    def exec_summary_box(self, text: str):
        """Render executive summary as a prominent box."""
        if not text:
            return
        chars_per_line = 80
        raw_lines = text.split("\n")
        total_lines = sum(max(1, (len(line) // chars_per_line) + 1) for line in raw_lines)
        h = max(30, total_lines * 5.0 + 18)

        self._need_page(h + 6)
        y = self.get_y()

        # Background
        self.set_fill_color(245, 248, 255)
        self.rect(MARG, y, CONTENT_W, h, "F")
        # Top accent bar
        self.set_fill_color(*ACCENT)
        self.rect(MARG, y, CONTENT_W, 2, "F")
        # Label
        self.set_xy(MARG + 4, y + 4)
        self.set_font(self._fn, "B", 8)
        self.set_text_color(*NAVY)
        self.cell(0, 4, _cl("Executive Summary"), ln=True)
        # Content
        self.set_xy(MARG + 4, y + 11)
        self.set_font(self._fn, "", 8.5)
        self.set_text_color(50, 65, 85)
        self.multi_cell(CONTENT_W - 8, 5.0, _cl(text))
        self.set_text_color(0, 0, 0)
        self.set_y(max(self.get_y() + 3, y + h + 3))

    # ── Glossary box ──
    def glossary_box(self, terms: list[tuple[str, str]]):
        """Abbreviation/term definitions in 2-column layout."""
        if not terms:
            return

        col_w = CONTENT_W / 2 - 2
        row_h = 5.2
        n_rows = (len(terms) + 1) // 2
        box_h = n_rows * row_h + 12

        self._need_page(box_h + 6)
        y0 = self.get_y()

        self.set_fill_color(245, 248, 252)
        self.rect(MARG, y0, CONTENT_W, box_h, "F")
        self.set_fill_color(*ACCENT)
        self.rect(MARG, y0, 2.5, box_h, "F")

        # Title
        self.set_xy(MARG + 5, y0 + 2)
        self.set_font(self._fn, "B", 6.5)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.cell(28, 3.5, "  Terms & Definitions", fill=True)

        # Items (2-column)
        self.set_text_color(40, 40, 40)
        for i, (abbr, defn) in enumerate(terms):
            col = i % 2
            row = i // 2
            x = MARG + 5 + col * (col_w + 2)
            y = y0 + 9 + row * row_h

            self.set_xy(x, y)
            self.set_font(self._fn, "B", 7)
            self.set_text_color(*ACCENT)
            self.cell(18, row_h, _cl(abbr))

            self.set_xy(x + 18, y)
            self.set_font(self._fn, "", 6.8)
            self.set_text_color(*SLATE)
            self.cell(col_w - 20, row_h, _cl(defn))

        self.set_text_color(0, 0, 0)
        self.set_y(y0 + box_h + 4)

    # ── Conclusion box ──
    def conclusion_box(self, text: str):
        """Render a conclusion/notes section."""
        if not text:
            return
        self._need_page(30)
        self.ln(3)
        y0 = self.get_y()

        chars_per_line = 85
        raw_lines = text.split("\n")
        total_lines = sum(max(1, (len(line) // chars_per_line) + 1) for line in raw_lines)
        h = max(20, total_lines * 4.8 + 12)

        self.set_fill_color(248, 250, 252)
        self.rect(MARG, y0, CONTENT_W, h, "F")
        self.set_fill_color(*NAVY)
        self.rect(MARG, y0, CONTENT_W, 1.5, "F")

        self.set_xy(MARG + 3, y0 + 4)
        self.set_font(self._fn, "", 8)
        self.set_text_color(50, 65, 85)
        self.multi_cell(CONTENT_W - 6, 4.8, _cl(text))
        self.set_text_color(0, 0, 0)
        self.set_y(max(self.get_y() + 3, y0 + h + 3))


# =====================================================================
# Chart builders (matplotlib -> PNG bytes)
# =====================================================================

def _build_transit_chart(transit_df: pd.DataFrame) -> Optional[bytes]:
    """Build transit time chart (bar for single day, line for multi-day)."""
    if transit_df.empty or "date" not in transit_df.columns:
        return None

    metrics = {
        "mat_minutes": ("MAT", _m(ACCENT)),
        "lbt_minutes": ("LBT", _m(AMBER)),
        "eod_minutes": ("EOD", (1.0, 0.55, 0.26)),
    }
    available = {k: v for k, v in metrics.items() if k in transit_df.columns}
    if not available:
        return None

    daily = transit_df.groupby("date").agg({
        col: lambda x: x.dropna().mean() for col in available
    }).reset_index().sort_values("date")

    n_days = len(daily)

    if n_days == 1:
        fig, ax = plt.subplots(figsize=(8, 2.0), facecolor="white")
        labels = [v[0] for v in available.values()]
        colors = [v[1] for v in available.values()]
        vals = [daily[col].iloc[0] if col in daily.columns and not pd.isna(daily[col].iloc[0]) else 0
                for col in available]
        bars = ax.barh(labels, vals, color=colors, height=0.5)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} min", va="center", fontsize=8)
        ax.set_xlabel("Minutes", fontsize=8)
        ax.set_xlim(0, max(vals) * 1.3 if vals else 30)
        ax.set_title("Transit Time Summary (MAT / LBT / EOD)", fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_bytes(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="white")
        for col, (label, color) in available.items():
            if col in daily.columns:
                ax.plot(range(n_days), daily[col], marker="o", markersize=4,
                        linewidth=1.5, color=color, label=label)
        ax.set_xticks(range(n_days))
        date_labels = [d[-4:] if len(d) == 8 else d for d in daily["date"]]
        ax.set_xticklabels(date_labels, fontsize=7, rotation=45)
        ax.set_ylabel("Minutes", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Daily Average Transit Time", fontsize=9, fontweight="bold")
        fig.tight_layout()
        return _fig_to_bytes(fig)


def _build_route_segment_chart(transit_df: pd.DataFrame) -> Optional[bytes]:
    """Build stacked horizontal bar chart of route segment times for all 4 transit types."""
    if transit_df.empty:
        return None

    transit_types = [
        {
            "label": "MAT (Exit)",
            "cols": ["seg_gate_to_outdoor", "seg_outdoor_to_hoist", "seg_hoist_to_fab"],
            "seg_labels": ["Gate->Outdoor", "Outdoor->Hoist", "Hoist->FAB"],
        },
        {
            "label": "LBT-Out",
            "cols": ["seg_lbt_out_fab_to_hoist", "seg_lbt_out_hoist_to_outdoor", "seg_lbt_out_outdoor_to_gate"],
            "seg_labels": ["FAB->Hoist", "Hoist->Outdoor", "Outdoor->Gate"],
        },
        {
            "label": "LBT-In",
            "cols": ["seg_lbt_in_gate_to_outdoor", "seg_lbt_in_outdoor_to_hoist", "seg_lbt_in_hoist_to_fab"],
            "seg_labels": ["Gate->Outdoor", "Outdoor->Hoist", "Hoist->FAB"],
        },
        {
            "label": "EOD",
            "cols": ["seg_eod_fab_to_hoist", "seg_eod_hoist_to_outdoor", "seg_eod_outdoor_to_gate"],
            "seg_labels": ["FAB->Hoist", "Hoist->Outdoor", "Outdoor->Gate"],
        },
    ]

    valid_types = []
    for tt in transit_types:
        if all(c in transit_df.columns for c in tt["cols"]):
            means = [transit_df[c].dropna().mean() for c in tt["cols"]]
            if any(not pd.isna(m) for m in means):
                valid_types.append((tt, means))

    if not valid_types:
        return None

    seg_colors = [_m(ACCENT), _m(AMBER), _m(GREEN)]

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(valid_types) * 0.8 + 0.5)), facecolor="white")

    y_labels = [vt[0]["label"] for vt in valid_types]
    y_pos = range(len(valid_types))

    for seg_idx in range(3):
        lefts = []
        widths = []
        for vt_data in valid_types:
            means = vt_data[1]
            prev_sum = sum(m if not pd.isna(m) else 0 for m in means[:seg_idx])
            w = means[seg_idx] if not pd.isna(means[seg_idx]) else 0
            lefts.append(prev_sum)
            widths.append(w)

        ax.barh(y_pos, widths, left=lefts, height=0.5,
                color=seg_colors[seg_idx], label=f"Seg {seg_idx+1}" if seg_idx < 3 else "")

        for i, w in enumerate(widths):
            if w > 0.5:
                ax.text(lefts[i] + w / 2, i, f"{w:.1f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Minutes", fontsize=8)
    ax.set_title("Route Segment Times by Transit Type", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _build_opr_chart(weekly_overall: pd.DataFrame) -> Optional[bytes]:
    """Build weekly overall OPR bar chart."""
    opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "avg_opr"
    week_col = "week_label" if "week_label" in weekly_overall.columns else "week"

    if weekly_overall.empty or opr_col not in weekly_overall.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="white")
    vals = weekly_overall[opr_col].values
    colors = [_m(GREEN) if v >= 0.7 else _m(AMBER) if v >= 0.5 else _m(RED) for v in vals]
    bars = ax.bar(range(len(vals)), vals * 100, color=colors, width=0.6)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(weekly_overall[week_col].tolist(), fontsize=7)
    ax.set_ylabel("OPR (%)", fontsize=8)
    ax.set_ylim(0, max(vals * 100) * 1.2 if len(vals) > 0 else 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Weekly Overall OPR", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _build_ewi_cre_combined(worker_df: pd.DataFrame) -> Optional[bytes]:
    """EWI + CRE histogram side-by-side."""
    has_ewi = not worker_df.empty and "ewi" in worker_df.columns
    has_cre = not worker_df.empty and "cre" in worker_df.columns
    if not has_ewi and not has_cre:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5), facecolor="white")

    if has_ewi:
        axes[0].hist(worker_df["ewi"].dropna(), bins=20,
                     color=_m(ACCENT), edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("EWI", fontsize=8)
        axes[0].set_ylabel("Workers", fontsize=8)
        axes[0].set_title("EWI Distribution", fontsize=9, fontweight="bold")
        axes[0].tick_params(labelsize=7)
        axes[0].grid(axis="y", alpha=0.3)
    else:
        axes[0].axis("off")

    if has_cre:
        axes[1].hist(worker_df["cre"].dropna(), bins=20,
                     color=_m(RED), edgecolor="white", alpha=0.85)
        axes[1].set_xlabel("CRE", fontsize=8)
        axes[1].set_ylabel("Workers", fontsize=8)
        axes[1].set_title("CRE Distribution", fontsize=9, fontweight="bold")
        axes[1].tick_params(labelsize=7)
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    fig.tight_layout(pad=1.5)
    return _fig_to_bytes(fig)


def _build_congestion_chart(space_df: pd.DataFrame) -> Optional[bytes]:
    """Build space congestion horizontal bar chart."""
    if space_df.empty:
        return None

    count_cols = [c for c in [
        "avg_workers", "max_workers", "avg_occupancy",
        "unique_workers", "total_person_minutes", "total_visits",
    ] if c in space_df.columns]
    if not count_cols:
        return None

    main_col = count_cols[0]

    _token_map = {
        "work_zone": "Work Zone (FAB)",
        "breakroom": "Break Room",
        "smoking_area": "Smoking Area",
        "restroom": "Restroom",
        "transit": "Hoist / Transit",
        "timeclock": "Time Clock (Gate)",
        "outdoor_work": "Outdoor Work",
        "unknown": "Unmapped",
    }

    if "locus_token" in space_df.columns:
        grp_col = "locus_token"
        agg = space_df.groupby(grp_col)[main_col].sum().sort_values()
        y_labels = [_token_map.get(str(k), str(k)) for k in agg.index]
    else:
        grp_col = next(
            (c for c in ["locus_name", "locus_id"] if c in space_df.columns), None
        )
        if grp_col is None:
            return None
        agg = space_df.groupby(grp_col)[main_col].mean().sort_values()
        y_labels = [str(k) for k in agg.index]

    n = len(agg)

    fig, ax = plt.subplots(figsize=(8, max(2.0, n * 0.45 + 0.8)), facecolor="white")
    bars = ax.barh(range(n), agg.values, color=_m(ACCENT), height=0.55)
    for bar, val in zip(bars, agg.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(main_col.replace("_", " ").title(), fontsize=8)
    ax.set_title("Space Occupancy by Location", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_bytes(fig)


# =====================================================================
# Parameter descriptions for each section
# =====================================================================

PARAM_DESC_TRANSIT = (
    "MAT (Morning Assembly Time): 출근 후 첫 작업구역 도달까지 소요 시간. 15분 초과 시 출입 병목 의심.\n"
    "LBT (Lunch Break Transit): 중식 외출 및 복귀 이동 시간. 요일별 변동 가능 (금요일 단축 경향).\n"
    "EOD (End-of-Day Transit): 마지막 작업구역 -> 출입구 이동 시간. 18분 초과 시 퇴근 혼잡 의심."
)

PARAM_DESC_ROUTE_SEGMENT = (
    "구간별 이동 시간: 출입(Gate) -> 야외(Outdoor) -> 호이스트(Hoist) -> FAB 작업층 경로의 각 구간 소요 시간.\n"
    "호이스트 구간이 길면 수직 이동 대기가 병목일 가능성."
)

PARAM_DESC_OPR = (
    "OPR (Operation Rate): 진동센서 기반 장비 실제 가동 비율. 70% 이상=양호, 50~70%=개선여지, 50% 미만=점검필요.\n"
    "BP (Business Partner): 협력업체별 장비 가동률. 공정 진행 단계에 따라 층별 편차 발생 가능."
)

PARAM_DESC_EWI_CRE = (
    "EWI (Engagement & Work Intensity): 진동센서 기반 작업 강도 (0~1). 0.7+=고집중, 0.4-=저집중.\n"
    "CRE (Cumulative Risk Exposure): 공간 위험+밀집도+개인 특성 통합 (0~1). 0.5+=고위험 주의."
)

PARAM_DESC_CONGESTION = (
    "공간 혼잡도: 각 공간 유형별 평균 점유 인원. 호이스트 혼잡은 MAT/LBT에 직접 영향.\n"
    "작업구역(FAB 층별) 혼잡은 층간 병목 또는 특정 공정 집중이 원인일 수 있음."
)


# =====================================================================
# Statistical fallback commentary (when LLM is unavailable)
# =====================================================================

def _fallback_transit_commentary(transit_df: pd.DataFrame) -> str:
    """Generate basic statistical summary for transit section."""
    if transit_df.empty:
        return ""
    parts = []
    for col, label, threshold in [
        ("mat_minutes", "MAT (출근 대기)", 15),
        ("lbt_minutes", "LBT (중식 이동)", 20),
        ("eod_minutes", "EOD (퇴근 이동)", 18),
    ]:
        if col in transit_df.columns:
            vals = transit_df[col].dropna()
            if not vals.empty:
                avg = vals.mean()
                med = vals.median()
                mx = vals.max()
                over = (vals > threshold).sum()
                pct = over / len(vals) * 100
                status = "정상 범위" if avg <= threshold else "기준 초과"
                parts.append(
                    f"{label}: 평균 {avg:.1f}분 (중앙값 {med:.1f}분, 최대 {mx:.1f}분). "
                    f"{threshold}분 초과 작업자 {over}명({pct:.0f}%). [{status}]"
                )
    if not parts:
        return ""
    return "[통계 요약]\n" + "\n".join(parts)


def _fallback_equipment_commentary(equip: dict) -> str:
    """Generate basic statistical summary for equipment section."""
    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    if weekly_overall.empty:
        return ""
    opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "avg_opr"
    if opr_col not in weekly_overall.columns:
        return ""
    vals = weekly_overall[opr_col]
    latest = vals.iloc[-1] if len(vals) > 0 else 0
    avg = vals.mean()
    trend = "상승" if len(vals) > 1 and vals.iloc[-1] > vals.iloc[-2] else "하락" if len(vals) > 1 and vals.iloc[-1] < vals.iloc[-2] else "유지"
    master = equip.get("master", pd.DataFrame())
    n_equip = len(master) if not master.empty else 0
    return (
        f"[통계 요약]\n"
        f"최근 주차 전체 가동률: {latest:.1%} (평균: {avg:.1%}). 추세: {trend}.\n"
        f"등록 장비 수: {n_equip}대."
    )


def _fallback_worker_commentary(worker_df: pd.DataFrame) -> str:
    """Generate basic statistical summary for worker section."""
    if worker_df.empty:
        return ""
    parts = []
    n = len(worker_df)
    if "ewi" in worker_df.columns:
        ewi = worker_df["ewi"].dropna()
        high_ewi = (ewi >= 0.7).sum()
        low_ewi = (ewi < 0.4).sum()
        parts.append(f"EWI 분포: 평균 {ewi.mean():.3f}, 고집중(>=0.7) {high_ewi}명, 저집중(<0.4) {low_ewi}명.")
    if "cre" in worker_df.columns:
        cre = worker_df["cre"].dropna()
        high_cre = (cre >= 0.5).sum()
        pct = high_cre / n * 100 if n > 0 else 0
        parts.append(f"CRE 분포: 평균 {cre.mean():.3f}, 고위험(>=0.5) {high_cre}명({pct:.1f}%).")
    if not parts:
        return ""
    return "[통계 요약]\n" + "\n".join(parts)


def _fallback_congestion_commentary(space_df: pd.DataFrame) -> str:
    """Generate basic statistical summary for congestion section."""
    if space_df.empty:
        return ""
    count_cols = [c for c in [
        "avg_workers", "max_workers", "avg_occupancy",
        "unique_workers", "total_person_minutes", "total_visits",
    ] if c in space_df.columns]
    if not count_cols:
        return ""
    main_col = count_cols[0]
    if "locus_token" in space_df.columns:
        top = space_df.groupby("locus_token")[main_col].sum().sort_values(ascending=False)
    elif "locus_name" in space_df.columns:
        top = space_df.groupby("locus_name")[main_col].sum().sort_values(ascending=False)
    else:
        return ""
    top3 = top.head(3)
    parts = [f"{k}: {v:.0f}" for k, v in top3.items()]
    return f"[통계 요약]\n점유도 상위 공간: {', '.join(parts)}."


def _fallback_exec_summary(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    equip: dict,
    n_days: int,
) -> str:
    """Generate basic executive summary from KPIs when LLM is unavailable."""
    parts = []
    if not worker_df.empty and "user_no" in worker_df.columns:
        n_workers = worker_df["user_no"].nunique()
        parts.append(f"분석 기간 {n_days}일, 고유 작업자 {n_workers}명.")
    if not worker_df.empty and "ewi" in worker_df.columns:
        avg_ewi = worker_df["ewi"].mean()
        parts.append(f"평균 작업강도(EWI) {avg_ewi:.3f}.")
    if not worker_df.empty and "cre" in worker_df.columns:
        avg_cre = worker_df["cre"].mean()
        high_cre = (worker_df["cre"] >= 0.5).sum()
        parts.append(f"평균 위험노출(CRE) {avg_cre:.3f}, 고위험 {high_cre}명.")
    if not transit_df.empty:
        for col, label in [("mat_minutes", "MAT"), ("lbt_minutes", "LBT"), ("eod_minutes", "EOD")]:
            if col in transit_df.columns:
                avg = transit_df[col].dropna().mean()
                parts.append(f"평균 {label} {avg:.1f}분.")
    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    if not weekly_overall.empty:
        opr_col = "opr_mean" if "opr_mean" in weekly_overall.columns else "avg_opr"
        if opr_col in weekly_overall.columns:
            latest = weekly_overall[opr_col].iloc[-1]
            parts.append(f"최근 장비 가동률 {latest:.1%}.")
    if not parts:
        return "데이터가 부족하여 요약을 생성할 수 없습니다."
    return " ".join(parts)


# =====================================================================
# Main PDF generation function
# =====================================================================

def generate_report_pdf(
    worker_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    company_df: pd.DataFrame,
    space_df: pd.DataFrame,
    equip: dict,
    selected_dates: list[str],
    period_label: str,
    report_mode: str,
    ai_summary: str = "",
    commentaries: dict[str, str] | None = None,
) -> bytes:
    """
    Generate a complete PDF report with AI-powered commentary.

    Args:
        worker_df: Worker metrics DataFrame
        transit_df: Transit time DataFrame
        company_df: Company metrics DataFrame
        space_df: Space congestion DataFrame
        equip: Equipment data dict (weekly_bp_opr, weekly_overall_opr, master)
        selected_dates: List of date strings (YYYYMMDD)
        period_label: Human-readable period string
        report_mode: Report type label
        ai_summary: Optional AI executive summary text
        commentaries: Dict of section -> AI commentary text
            Keys: "exec_summary", "transit", "equipment", "worker", "congestion", "conclusion"

    Returns:
        PDF file as bytes
    """
    pdf = DeepConPDF()
    pdf.alias_nb_pages()
    n_days = len(selected_dates)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    c = commentaries or {}

    # ── 1. Cover ──
    pdf.add_cover(
        title="Deep Con at M15X",
        period=f"{period_label} ({n_days} days)",
        generated_at=generated_at,
    )

    # ── 2. Executive Summary ──
    pdf.add_page()
    pdf.section_title(1, "Executive Summary")

    exec_text = c.get("exec_summary", "")
    if not exec_text:
        exec_text = _fallback_exec_summary(worker_df, transit_df, equip, n_days)
    pdf.exec_summary_box(exec_text)

    pdf.separator()

    # ── 3. KPI Summary + Glossary ──
    pdf.section_title(2, "KPI Summary")

    pdf.glossary_box([
        ("MAT",  "Morning Assembly Time -- 출근 후 작업 위치 도착까지 소요 시간"),
        ("LBT",  "Lunch Break Transit -- 중식 외출 및 복귀 이동 시간 (왕복)"),
        ("EOD",  "End-of-Day Transit -- 작업 종료 후 게이트 퇴장까지 소요 시간"),
        ("EWI",  "Engagement & Work Intensity -- 작업 강도 지수 (0~1, 높을수록 활발)"),
        ("CRE",  "Cumulative Risk Exposure -- 누적 위험 노출 지수 (0~1, 0.5 이상 고위험)"),
        ("OPR",  "Operation Rate -- 장비 가동률 (가동 시간 / 총 근무 시간)"),
        ("BP",   "Business Partner -- 협력업체 (하도급 업체)"),
        ("FAB",  "Fabrication Building -- 반도체 제조 공장 건물 (5F~RF)"),
        ("SII",  "Safety Isolation Index -- 단독 고위험 구역 체류 지수"),
        ("GAP",  "Gap in Activity Pattern -- BLE 신호 공백 구간 (미측정 시간)"),
    ])

    if not worker_df.empty:
        has_date_col = "date" in worker_df.columns
        if has_date_col and "user_no" in worker_df.columns:
            daily_workers = worker_df.groupby("date")["user_no"].nunique()
            avg_workers = daily_workers.mean()
        elif "user_no" in worker_df.columns:
            avg_workers = worker_df["user_no"].nunique()
        else:
            avg_workers = 0
        total_unique = worker_df["user_no"].nunique() if "user_no" in worker_df.columns else 0
        avg_ewi = worker_df["ewi"].mean() if "ewi" in worker_df.columns else 0
        avg_cre = worker_df["cre"].mean() if "cre" in worker_df.columns else 0

        ewi_color = GREEN if avg_ewi >= 0.6 else AMBER if avg_ewi >= 0.2 else RED
        cre_color = RED if avg_cre >= 0.7 else AMBER if avg_cre >= 0.4 else GREEN

        pdf.kpi_row([
            ("Avg Daily Workers", f"{avg_workers:.0f}", ACCENT),
            ("Unique Workers", f"{total_unique:,}", ACCENT),
            ("Avg EWI", f"{avg_ewi:.3f}", ewi_color),
            ("Avg CRE", f"{avg_cre:.3f}", cre_color),
        ])

    if not transit_df.empty:
        transit_cards = []
        for col, label in [
            ("mat_minutes", "Avg MAT"),
            ("lbt_minutes", "Avg LBT"),
            ("eod_minutes", "Avg EOD"),
        ]:
            if col in transit_df.columns:
                val = transit_df[col].dropna().mean()
                transit_cards.append((label, f"{val:.1f} min", ACCENT))
        if transit_cards:
            pdf.kpi_row(transit_cards)

    pdf.separator()

    # ── 4. Transit Time Analysis ──
    pdf.section_title(3, "Transit Time Analysis (MAT/LBT/EOD)")

    transit_commentary = c.get("transit", "") or _fallback_transit_commentary(transit_df)

    transit_chart = _build_transit_chart(transit_df)
    if transit_chart:
        pdf.add_chart_with_commentary(
            transit_chart,
            commentary=transit_commentary,
            param_description=PARAM_DESC_TRANSIT,
        )

    # Route segment chart
    route_chart = _build_route_segment_chart(transit_df)
    if route_chart:
        pdf.ln(2)
        pdf.sub_title("Route Segment Analysis")
        pdf.add_chart_with_commentary(
            route_chart,
            param_description=PARAM_DESC_ROUTE_SEGMENT,
        )

    # Statistics table
    if not transit_df.empty:
        stats_rows = []
        for col, label in [
            ("mat_minutes", "MAT"),
            ("lbt_minutes", "LBT"),
            ("eod_minutes", "EOD"),
        ]:
            if col in transit_df.columns:
                vals = transit_df[col].dropna()
                if not vals.empty:
                    stats_rows.append([
                        label,
                        f"{vals.mean():.1f}",
                        f"{vals.median():.1f}",
                        f"{vals.min():.1f}",
                        f"{vals.max():.1f}",
                        f"{vals.std():.1f}",
                    ])
        if stats_rows:
            pdf.sub_title("Detailed Statistics")
            pdf.simple_table(
                ["Metric", "Mean", "Median", "Min", "Max", "StdDev"],
                stats_rows,
                col_widths=[35, 27, 27, 27, 27, 39],
            )

    pdf.separator()

    # ── 5. Equipment OPR ──
    pdf.section_title(4, "Table Lift OPR (Equipment)")

    equipment_commentary = c.get("equipment", "") or _fallback_equipment_commentary(equip)

    weekly_overall = equip.get("weekly_overall_opr", pd.DataFrame())
    weekly_bp = equip.get("weekly_bp_opr", pd.DataFrame())
    master = equip.get("master", pd.DataFrame())

    opr_chart = _build_opr_chart(weekly_overall) if not weekly_overall.empty else None
    if opr_chart:
        pdf.add_chart_with_commentary(
            opr_chart,
            commentary=equipment_commentary,
            param_description=PARAM_DESC_OPR,
        )

    if not weekly_bp.empty:
        bp_col = "bp_name" if "bp_name" in weekly_bp.columns else "company_name"
        opr_col = "opr_mean" if "opr_mean" in weekly_bp.columns else "avg_opr"
        week_col = "week_label" if "week_label" in weekly_bp.columns else "week"

        if bp_col in weekly_bp.columns and opr_col in weekly_bp.columns:
            latest_week = weekly_bp[week_col].max() if week_col in weekly_bp.columns else None
            if latest_week:
                latest = weekly_bp[weekly_bp[week_col] == latest_week]
                bp_rows = []
                for _, row in latest.sort_values(opr_col, ascending=False).head(10).iterrows():
                    bp_rows.append([
                        str(row.get(bp_col, "")),
                        str(row.get("equipment_count", "")),
                        f"{row.get(opr_col, 0):.1%}",
                    ])
                if bp_rows:
                    pdf.sub_title(f"BP OPR Summary - {latest_week}")
                    pdf.simple_table(
                        ["BP (Company)", "Equipment", "Avg OPR"],
                        bp_rows,
                        col_widths=[90, 40, 52],
                    )

    if not master.empty:
        pdf.text_block(f"Total registered equipment: {len(master)}")

    pdf.separator()

    # ── 6. Worker Analysis ──
    pdf.section_title(5, "Worker Analysis")

    worker_commentary = c.get("worker", "") or _fallback_worker_commentary(worker_df)

    ewi_cre_chart = _build_ewi_cre_combined(worker_df)
    if ewi_cre_chart:
        pdf.add_chart_with_commentary(
            ewi_cre_chart,
            commentary=worker_commentary,
            param_description=PARAM_DESC_EWI_CRE,
        )

    # Company breakdown
    if not company_df.empty and "company_name" in company_df.columns:
        count_col = "worker_count" if "worker_count" in company_df.columns else "n_workers"
        if count_col in company_df.columns:
            company_agg = company_df.groupby("company_name")[count_col].sum().reset_index()
            company_agg.columns = ["Company", "Workers"]
            top10 = company_agg.nlargest(10, "Workers")
            rows = [[str(r["Company"]), str(r["Workers"])] for _, r in top10.iterrows()]
            if rows:
                pdf.sub_title("Top 10 Companies by Worker Count")
                pdf.simple_table(["Company", "Workers (cumulative)"], rows,
                                 col_widths=[120, 62])

    # High-risk workers
    if "cre" in worker_df.columns:
        high_cre = worker_df[worker_df["cre"] >= 0.5]
        if not high_cre.empty:
            pct = len(high_cre) / len(worker_df) * 100
            pdf.text_block(
                f"High-risk workers (CRE >= 0.5): {len(high_cre)} / {len(worker_df)} ({pct:.1f}%)"
            )

    pdf.separator()

    # ── 7. Space Congestion ──
    pdf.section_title(6, "Space Congestion")

    congestion_commentary = c.get("congestion", "") or _fallback_congestion_commentary(space_df)

    congestion_chart = _build_congestion_chart(space_df)
    if congestion_chart:
        pdf.add_chart_with_commentary(
            congestion_chart,
            commentary=congestion_commentary,
            param_description=PARAM_DESC_CONGESTION,
        )
    elif space_df.empty:
        pdf.text_block("No congestion data available.")

    pdf.separator()

    # ── 8. Conclusion / Notes ──
    pdf.section_title(7, "Conclusion & Notes")

    conclusion_text = c.get("conclusion", "")
    if not conclusion_text:
        conclusion_text = (
            "본 리포트는 M15X FAB 건설현장의 BLE/진동센서 데이터를 기반으로 자동 생성되었습니다.\n"
            "BLE 신호 특성상 평균 50%의 음영(GAP)이 존재하며, 이는 건설현장의 정상적인 범위입니다.\n"
            "MAC 랜덤화로 인해 인원수는 T-Ward 기준이며, 실제 현장 인원과 차이가 있을 수 있습니다.\n"
            "수치는 참고용이며, 현장 상황과 함께 종합적으로 판단하시기 바랍니다."
        )
    pdf.conclusion_box(conclusion_text)

    # Legacy AI summary (from session state, if provided separately)
    if ai_summary and "exec_summary" not in c:
        pdf.ln(3)
        pdf.ai_box(ai_summary)

    # ── Output ──
    return bytes(pdf.output())
