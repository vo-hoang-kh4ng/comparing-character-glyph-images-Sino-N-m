"""Sinh báo cáo nộp thầy: `report/BaoCao.docx` + `report/BaoCao.pdf`.

Vì sao là script chứ không phải file .docx gõ tay
--------------------------------------------------
Yêu cầu của thầy là người đọc **tái lập được**. Một file .docx gõ tay thì con số trong đó không có
đường nào truy ngược về phép đo; sửa code xong quên sửa báo cáo là chuyện chắc chắn xảy ra. Ở đây
mọi con số nằm trong ``NUMBERS`` ngay đầu file, kèm chú thích lệnh nào sinh ra nó, nên soát lại là
việc đọc một khối duy nhất.

Vì sao .docx chứ không phải LaTeX
---------------------------------
Máy phát triển không có TeX Live, không có pandoc, không có quyền sudo — nên `report/report.tex`
(bản kỹ thuật dài) **chưa bao giờ biên dịch được**. LibreOffice thì có sẵn, và nó xuất được cả .docx
lẫn .pdf, đúng hai định dạng thầy nhận. Font Nôm phải cài vào ``~/.fonts`` trước, nếu không mọi chữ
Nôm ngoài BMP ra ô vuông rỗng — xem ``ensure_fonts``.

Độ dài
------
Nhắm 8-10 trang thân bài như một bài báo. Phần lệnh tái lập đầy đủ nằm ở phụ lục, và `BENCHMARK.md`
giữ bản dài (mọi bảng phụ, mọi lần chạy) — báo cáo trỏ sang đó chứ không chép lại.

Dùng
----
    .venv/bin/python build_report.py            # -> report/BaoCao.docx + .pdf
    .venv/bin/python build_report.py --no-pdf   # bỏ bước LibreOffice
"""

import argparse
import os
import re
import shutil
import subprocess

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

#: Font cho chữ Hán-Nôm. BabelStone phủ phần lớn; HanaMinB phủ Ext-B (chữ Nôm riêng).
NOM_FONT = "BabelStone Han"
MONO_FONT = "DejaVu Sans Mono"
BODY_FONT = "Times New Roman"

#: Hình học trang, dùng để co bảng cho vừa. A4 rộng 21 cm, lề 1,7 cm mỗi bên, khe giữa hai cột
#: 0,7 cm. Sai mấy con số này thì bảng tràn ra ngoài và bị cắt trong PDF.
TEXT_WIDTH_CM = 21.0 - 2 * 1.7
COLUMN_GAP_CM = 0.7
COLUMN_WIDTH_CM = (TEXT_WIDTH_CM - COLUMN_GAP_CM) / 2

#: Đường dẫn font cần có trong ~/.fonts để LibreOffice vẽ được chữ Nôm.
FONT_FILES = ("BabelStoneHan.ttf", "HanaMinA.otf", "HanaMinB.otf")


def ensure_fonts(font_dir="fonts"):
    """Chép font Nôm vào ~/.fonts. Không có bước này thì mọi chữ Ext-B ra ô vuông rỗng trong PDF.

    Lỗi này *im lặng* — .docx vẫn mở được, chỉ là chữ mất — nên phải làm chủ động chứ không đợi
    thấy hỏng mới sửa.
    """
    target = os.path.expanduser("~/.fonts")
    os.makedirs(target, exist_ok=True)
    copied = []
    for name in FONT_FILES:
        src = os.path.join(font_dir, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(target, name)):
            shutil.copy2(src, target)
            copied.append(name)
    if copied:
        subprocess.run(["fc-cache", "-f"], capture_output=True)
    return copied


# --- tiện ích dựng tài liệu ------------------------------------------------------------------

def _style_run(run, bold=False, mono=False, nom=False, size=None, italic=False):
    run.bold, run.italic = bold, italic
    name = MONO_FONT if mono else (NOM_FONT if nom else BODY_FONT)
    run.font.name = name
    # python-docx chỉ đặt w:ascii; chữ Hán rơi vào "East Asian" nên phải đặt riêng, không thì Word
    # lẫn LibreOffice đều thay bằng font mặc định và ra ô vuông.
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size:
        run.font.size = Pt(size)
    if mono:
        run.font.size = Pt((size or 10.5) - 1)


#: Ký hiệu đánh dấu inline. ``**đậm**``, ``*nghiêng*``, ``` `mã` ```, ``〔chữ Nôm〕``.
#: Ngoặc 〔〕 cố tình chọn ký tự không bao giờ xuất hiện tự nhiên trong văn bản tiếng Việt.
_MARKUP = re.compile(r"(\*\*|\*|`[^`]+`|〔[^〕]+〕)")


def rich(paragraph, text, size=None):
    """Đổ text có đánh dấu nhẹ vào paragraph.

    Đậm/nghiêng là **trạng thái bật tắt** chứ không phải cặp ngoặc bao quanh, nên ``**a `b` c**``
    giữ được định dạng đậm cho cả đoạn mã bên trong. Bản đầu tiên tách bằng một regex cặp và làm mất
    markup lồng nhau — chỉ lộ ra khi đọc bản PDF đã dựng.
    """
    bold = italic = False
    for piece in _MARKUP.split(text):
        if not piece:
            continue
        if piece == "**":
            bold = not bold
        elif piece == "*":
            italic = not italic
        elif piece.startswith("`"):
            _style_run(paragraph.add_run(piece[1:-1]), bold=bold, italic=italic, mono=True, size=size)
        elif piece.startswith("〔"):
            _style_run(paragraph.add_run(piece[1:-1]), bold=bold, italic=italic, nom=True,
                       size=(size or 11) + 2)
        else:
            _style_run(paragraph.add_run(piece), bold=bold, italic=italic, size=size)


class Report:
    """Bố cục hai cột kiểu bài hội nghị.

    ``python-docx`` không có API cho số cột, nên phải đặt thẳng ``w:num`` trên phần tử ``w:cols``
    của mỗi section. Đổi số cột giữa chừng = chèn một section liên tục (``CONTINUOUS``) mới; đó là
    cách duy nhất cho một bảng rộng tràn ngang cả trang rồi quay lại hai cột.
    """

    def __init__(self, columns=2):
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        for attr, value in (("top_margin", 2.0), ("bottom_margin", 2.0),
                            ("left_margin", 1.7), ("right_margin", 1.7)):
            setattr(section, attr, Cm(value))
        style = self.doc.styles["Normal"]
        style.font.name, style.font.size = BODY_FONT, Pt(9.5)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.05
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._num = 0
        self.body_columns = columns
        self._set_columns(section, 1, COLUMN_GAP_CM)   # phần đầu (tiêu đề + tóm tắt) luôn tràn ngang

    @staticmethod
    def _set_columns(section, num, gap_cm=0.7):
        cols = section._sectPr.xpath("./w:cols")[0]
        # Xoá mọi <w:col> con: nếu còn, Word ưu tiên bề rộng thủ công đó và bỏ qua w:num.
        for child in list(cols):
            cols.remove(child)
        cols.set(qn("w:num"), str(num))
        cols.set(qn("w:space"), str(int(gap_cm * 567)))

    def columns(self, num):
        """Chuyển sang bố cục `num` cột kể từ đây."""
        self._set_columns(self.doc.add_section(WD_SECTION.CONTINUOUS), num, COLUMN_GAP_CM)

    def body(self):
        """Về lại số cột của thân bài."""
        self.columns(self.body_columns)

    def title(self, text, subtitle=None, authors=None):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
        _style_run(p.add_run(text), bold=True, size=16)
        if subtitle:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            _style_run(p.add_run(subtitle), size=11, italic=True)
        if authors:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            _style_run(p.add_run(authors), size=9.5)

    def abstract(self, text, keywords=None, indent_cm=1.4):
        """Tóm tắt thụt lề hai bên, tràn ngang trang — quy ước của bài hội nghị."""
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(indent_cm)
        p.paragraph_format.right_indent = Cm(indent_cm)
        p.paragraph_format.space_after = Pt(4)
        _style_run(p.add_run("Tóm tắt—"), bold=True, italic=True, size=9)
        rich(p, text, size=9)
        if keywords:
            k = self.doc.add_paragraph()
            k.paragraph_format.left_indent = Cm(indent_cm)
            k.paragraph_format.right_indent = Cm(indent_cm)
            k.paragraph_format.space_after = Pt(12)
            _style_run(k.add_run("Từ khoá—"), bold=True, italic=True, size=9)
            _style_run(k.add_run(keywords), italic=True, size=9)

    def h(self, text, level=1):
        # Mỗi tiêu đề mở một danh sách đánh số mới. Style "List Number" của Word dùng chung một
        # numId cho cả tài liệu nên nó đếm tiếp từ danh sách trước — mục 8 từng bắt đầu ở số 5 vì
        # mục 1 đã dùng 1..4. Tự đánh số là cách chắc chắn nhất, không phụ thuộc numbering.xml.
        self._num = 0
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        _style_run(p.add_run(text), bold=True, size=10.5 if level == 1 else 9.8)

    def p(self, text, size=None):
        rich(self.doc.add_paragraph(), text, size)

    def bullet(self, text, numbered=False):
        if numbered:
            self._num += 1
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.95)
            p.paragraph_format.first_line_indent = Cm(-0.95)
            _style_run(p.add_run(f"{self._num}.  "), bold=False)
        else:
            p = self.doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.1
        rich(p, text)

    def code(self, lines):
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.0
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for i, line in enumerate(lines):
            if i:
                p.add_run().add_break()
            _style_run(p.add_run(line), mono=True, size=8.5)

    def table(self, header, rows, caption=None, widths=None, wide=False):
        """`wide=True` cho bảng tràn ngang cả trang rồi tự quay lại thân bài.

        Bảng 6-7 cột không đọc được trong một cột rộng 8 cm; đó là lý do có tham số này chứ không
        phải để trang trí.
        """
        if wide:
            self.columns(1)
        if caption:
            p = self.doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            _style_run(p.add_run(caption), italic=True, size=8.5)
        table = self.doc.add_table(rows=1, cols=len(header))
        table.style = "Table Grid"
        for cell, text in zip(table.rows[0].cells, header):
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
            rich(cell.paragraphs[0], f"**{text}**", size=8.5)
        for row in rows:
            cells = table.add_row().cells
            for cell, text in zip(cells, row):
                cell.paragraphs[0].paragraph_format.space_after = Pt(2)
                rich(cell.paragraphs[0], str(text), size=8.5)
        # Co bề rộng cột cho vừa khung chứa. Không có bước này thì bảng khai báo rộng hơn cột sẽ bị
        # CẮT MẤT cột cuối trong PDF — hỏng âm thầm, .docx mở ra vẫn thấy đủ.
        if widths:
            avail = TEXT_WIDTH_CM if wide else COLUMN_WIDTH_CM
            scale = avail / sum(widths)
            table.autofit = False
            scaled = [Cm(w * scale) for w in widths]
            # Phải set CẢ ``w:tblGrid`` (qua table.columns) lẫn bề rộng từng ô. Chỉ set ô thì
            # ``tblW`` vẫn là "auto" và LibreOffice dựng bảng theo tblGrid cũ — rộng cả trang, nên
            # cột cuối bị cắt mất khi bảng nằm trong một cột hẹp. Lỗi này im lặng: mở .docx vẫn đủ.
            for column, width in zip(table.columns, scaled):
                column.width = width
            for row in table.rows:
                for cell, width in zip(row.cells, scaled):
                    cell.width = width
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        if wide:
            self.body()
        return table

    def image(self, path, width_cm, caption=None):
        if not os.path.exists(path):
            return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Cm(width_cm))
        if caption:
            c = self.doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _style_run(c.add_run(caption), italic=True, size=9.5)

    def pagebreak(self):
        self.doc.add_page_break()

    def save(self, path):
        self.doc.save(path)


#: Chỉ liệt kê những tài liệu nhóm thực sự dùng và **kiểm được thông tin thư mục**. Các bộ dữ liệu
#: thư pháp nêu ở mục 9 (MCCD, NomNaOCR, IHR-NomDB) cố ý KHÔNG đưa vào đây: nhóm mới đọc mô tả chứ
#: chưa đối chiếu bản gốc, và một mục tham khảo bịa thông tin còn tệ hơn là thiếu.
REFERENCES = [
    "A. Yang và cộng sự, \"Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese\", "
    "arXiv:2211.01335, 2022.",
    "M. Oquab và cộng sự, \"DINOv2: Learning Robust Visual Features without Supervision\", "
    "arXiv:2304.07193, 2023.",
    "K. He, X. Zhang, S. Ren và J. Sun, \"Deep Residual Learning for Image Recognition\", CVPR, 2016.",
    "P. Khosla và cộng sự, \"Supervised Contrastive Learning\", NeurIPS, 2020.",
    "J. Deng, J. Guo, N. Xue và S. Zafeiriou, \"ArcFace: Additive Angular Margin Loss for Deep Face "
    "Recognition\", CVPR, 2019.",
    "J. Johnson, M. Douze và H. Jégou, \"Billion-scale similarity search with GPUs\", "
    "IEEE Transactions on Big Data, 2019.",
    "Y. A. Malkov và D. A. Yashunin, \"Efficient and robust approximate nearest neighbor search "
    "using Hierarchical Navigable Small World graphs\", IEEE TPAMI, 2020.",
    "E. B. Wilson, \"Probable Inference, the Law of Succession, and Statistical Inference\", "
    "Journal of the American Statistical Association, 1927.",
]


# --- nội dung --------------------------------------------------------------------------------
#
# MỌI CON SỐ dưới đây đến từ các lệnh ở Phụ lục A. Không có số nào ước lượng bằng tay. Nếu sửa code
# rồi chạy lại, sửa ở đây là đủ — không có bản sao nào khác trong file này.

def build(rp):
    rp.title(
        "Tìm kiếm ảnh ký tự Hán-Nôm tương đồng bằng học tương phản trên cặp dương sinh từ font",
        "Đồ án môn học — mã nguồn, dữ liệu, mô hình và lệnh tái lập kèm theo")

    rp.abstract(
        "Báo cáo trình bày một hệ thống tìm kiếm ảnh ký tự Hán-Nôm tương đồng trên corpus 26.044 "
        "glyph, và quá trình cải thiện nó qua ba bước đo được. **Bước một**: thay bộ trích đặc trưng "
        "ResNet18 của bản gốc bằng tháp ảnh Chinese-CLIP, nâng radical@k từ 0,2316 lên 0,5361 "
        "(2,31 lần). **Bước hai**: xây dựng phép đánh giá trên ảnh scan thật thay vì chỉ dựa vào "
        "chỉ số proxy — 59 ảnh viết tay, nhãn giải mã từ tên file rồi được người đọc kiểm lại; trên "
        "tập này mô hình tốt nhất chỉ đạt hit@1 = 0,0847, thấp hơn hẳn mức mà chỉ số proxy gợi ý. "
        "**Bước ba**: fine-tune bằng học tương phản có giám sát (SupCon) trên 141.981 ảnh render đa "
        "font, nhãn là mã Unicode, đưa hit@1 trên chính 59 ảnh scan đó lên **0,5932** — hai khoảng "
        "tin cậy 95% rời hẳn nhau. Trên bài toán phụ (xếp hạng trong nhóm ký tự cùng âm Quốc Ngữ), "
        "độ chính xác top-1 tăng từ 0,4746 lên **0,8136**. Các lớp Unicode bị giữ lại hoàn toàn "
        "ngoài tập huấn luyện đạt kết quả bằng đúng các lớp đã huấn luyện (0,9939 so với 0,9954), "
        "nên phần cải thiện không đến từ việc ghi nhớ. Chúng tôi cũng khảo sát một tầng xếp hạng lại "
        "bằng siêu dữ liệu ngôn ngữ học và báo cáo nó như một **kết quả âm**: metadata có trần cao "
        "(oracle +17,5 điểm) nhưng suy ra được siêu dữ liệu của ảnh truy vấn mới là chỗ nghẽn. "
        "Báo cáo giữ lại đầy đủ các kết quả âm — trong đó có một lần huấn luyện bị sụp đổ biểu diễn "
        "— vì nguyên nhân của chúng có tính lặp lại.",
        keywords="truy hồi ảnh, chữ Nôm, học tương phản, SupCon, Chinese-CLIP, xếp hạng lại, "
                 "kết quả âm")

    rp.body()

    # ---------------------------------------------------------------- 1
    rp.h("1. Bài toán")
    rp.p(
        "Cho một ảnh glyph Hán-Nôm, tìm những ký tự trong corpus có hình dạng gần nó nhất. Bài toán "
        "được chia làm hai phần, tương ứng hai tình huống sử dụng khác nhau:")
    rp.bullet(
        "**Phần 1 — tìm trên toàn corpus.** Không biết trước ký tự là gì; xếp hạng toàn bộ 26.044 "
        "ứng viên. Đây là tình huống khó, dùng khi số hoá một trang chữ chưa được phiên âm.")
    rp.bullet(
        "**Phần 2 — xếp hạng trong nhóm cùng âm Quốc Ngữ.** Đã biết chữ đó đọc là gì (ví dụ "
        "\"trăm\"), cần chọn đúng tự dạng trong số các ứng viên cùng âm — trung vị 7 ứng viên. Đây "
        "là tình huống thường gặp khi hiệu đính một bản phiên âm đã có.")
    rp.p("Đóng góp của nhóm, theo thứ tự giá trị:")
    rp.bullet(
        "Một **phép đánh giá trên dữ liệu thật** thay cho chỉ số proxy: 59 ảnh scan viết tay có "
        "nhãn, kèm thang tin cậy do người gán độc lập với mọi điểm số của mô hình (mục 4).",
        numbered=True)
    rp.bullet(
        "**Fine-tune** nâng `hit@1` trên scan thật từ 0,0847 lên 0,5932, kèm phép kiểm chứng minh "
        "kết quả không phải do ghi nhớ (mục 5.4, 5.5).", numbered=True)
    rp.bullet(
        "Một **quy trình sinh dữ liệu huấn luyện** từ corpus vốn chỉ có một ảnh mỗi ký tự, bằng "
        "render đa font cộng mô hình suy giảm ảnh được hiệu chỉnh theo số đo của scan thật "
        "(mục 3.4).", numbered=True)
    rp.bullet(
        "**Bốn kết quả âm** được ghi lại đầy đủ, gồm phân tích trần của chỉ số `radical@k` và một "
        "lần sụp đổ biểu diễn khi huấn luyện (mục 6).", numbered=True)

    # ---------------------------------------------------------------- 2
    rp.h("2. Dữ liệu")
    rp.table(
        ["Tập", "Kích thước", "Vai trò"],
        [["`images/`", "26.044 ảnh JPG 70×70", "Corpus tra cứu. Mỗi ký tự đúng **một** ảnh, tên file là mã Unicode."],
         ["`final_characteristics-v2.xlsx`", "31.208 dòng", "Bộ thủ, số nét, phân rã IDS. 5.164 dòng không có ảnh nên bị loại."],
         ["`QuocNgu_SinoNom_Dic.xlsx`", "—", "Ánh xạ âm Quốc Ngữ ↔ ký tự Sino-Nôm, dùng cho Phần 2."],
         ["`test_images/`", "59 ảnh scan", "**Tập đánh giá thật.** Chữ viết tay, kích thước không đồng nhất (88×96 … 178×162)."],
         ["`output/rendered/`", "131.604 ảnh", "Sinh ra để huấn luyện: mỗi ký tự render qua tối đa 7 font."]],
        caption="Bảng 1. Các tập dữ liệu. Ba tập đầu là đầu vào cho trước; hai tập sau do nhóm tạo hoặc gán nhãn.",
        widths=[5.0, 3.4, 8.2], wide=True)
    rp.p(
        "Đặc điểm quyết định toàn bộ thiết kế: **corpus chỉ có một ảnh cho mỗi ký tự.** Học metric "
        "cần cặp dương — hai ảnh khác nhau của cùng một lớp — nên trong dữ liệu gốc *không tồn tại "
        "cặp nào*. Mục 3.4 giải quyết đúng chỗ này.")
    rp.p(
        "Ảnh trong `test_images/` được đặt tên bằng kiểu gõ Telex (`cofn.jpg` = \"còn\", "
        "`nguwowsi.jpg` = \"người\"), nên giải mã ngược ra chữ Quốc Ngữ rồi tra từ điển sẽ được "
        "**tập ký tự đúng** cho ảnh đó. Đây là nhãn thật, không phải proxy — và là lý do tập 59 ảnh "
        "này có giá trị lớn hơn nhiều so với kích thước của nó.")

    # ---------------------------------------------------------------- 3
    rp.h("3. Phương pháp")

    rp.h("3.1. Trích đặc trưng và tìm kiếm", 2)
    rp.p(
        "Mọi backend đưa ảnh về một vector đã chuẩn hoá L2, nên tích vô hướng chính là cosine. "
        "Tìm kiếm dùng Faiss `IndexFlatIP` — tìm chính xác, không xấp xỉ (lý do ở mục 5.3). Kết quả "
        "trích đặc trưng được cache ra `.npz` nên chỉ trả chi phí một lần.")
    rp.p(
        "Bản gốc dùng ResNet18 pretrain ImageNet, thay lớp `conv1` bằng một lớp 1 kênh khởi tạo "
        "ngẫu nhiên. Nhóm giữ nguyên bản này **nguyên vẹn từng bit** làm mốc \"trước\", và thêm năm "
        "backend để so: một biến thể `resnet18-gray` sửa đúng một dòng, ba mô hình Chinese-CLIP ba "
        "cỡ, và DINOv2.")

    rp.h("3.2. Phần 2 — xếp hạng trong nhóm cùng âm", 2)
    rp.p(
        "Với một âm Quốc Ngữ, từ điển cho ra tập ứng viên; hệ thống xếp hạng chúng theo độ giống "
        "ảnh query. Hai cách chấm điểm được so sánh: **histogram** (đặc trưng thủ công của bản gốc) "
        "và **embedding** (dùng lại vector của Phần 1). Trên phép thử tự truy hồi, embedding đạt "
        "64/64 còn histogram 62/64; đáng chú ý hơn, histogram nén cả nhóm ứng viên vào một dải điểm "
        "rộng 0,003, tức nó gần như không phân biệt được gì bên trong nhóm.")

    rp.h("3.3. Sinh cặp dương bằng render đa font", 2)
    rp.p(
        "Vì mỗi ký tự chỉ có một ảnh, nhóm render lại từng mã Unicode qua bảy font Hán-Nôm khác "
        "nhau. Cùng một lớp, khác kiểu chữ — đúng loại cặp dương cần để dạy mô hình bỏ qua khác "
        "biệt tự dạng. Độ phủ được kiểm bằng bảng `cmap` của từng font chứ không phỏng đoán: "
        "hanamin 99,9%, babelstone 88,3%, iming 83,7%, lxgw-kai 71,3%, noto-serif và noto-sans "
        "59,9%, zhuque-fangsong 42,3%; hợp nhất lại phủ 26.033/26.044 ký tự.")
    rp.p(
        "**Ký tự không có trong `cmap` bị bỏ qua, không render.** Nếu render bừa, font trả về ô "
        "tofu — một ô vuông rỗng giống hệt nhau ở mọi ký tự — đưa vào huấn luyện sẽ đầu độc cặp "
        "dương.")
    rp.p(
        "Nhưng tập render sạch **quá dễ**: ảnh render tự truy hồi đúng chữ ở `hit@1` = 0,7018 trong "
        "khi scan thật chỉ đạt 0,0847, chênh gần chín lần. Nghĩa là phần lớn khoảng cách tới thực "
        "tế **không nằm ở font**. Nhóm đo trực tiếp sáu chỉ số ảnh trên hai tập để tìm chỗ lệch:")
    rp.table(
        ["Chỉ số", "Scan thật", "Render", "Tỉ lệ", "Ý nghĩa"],
        [["edge", "15,6", "102,3", "0,15×", "scan mờ hơn nhiều — khoảng cách lớn nhất"],
         ["contrast", "91,6", "232,8", "0,39×", "mực xám nhạt, không đen tuyền"],
         ["thick", "0,051", "0,028", "1,82×", "nét scan dày gần gấp đôi"],
         ["fill", "0,349", "0,206", "1,69×", "hệ quả của nét dày"],
         ["noise", "10,1", "2,7", "3,77×", "hạt nhiễu giấy và cảm biến"],
         ["bg", "254,0", "255,0", "≈1×", "nền giấy hơi ngà"]],
        caption="Bảng 2. Khoảng cách đo được giữa scan thật và ảnh render. Hàm suy giảm mô phỏng đúng sáu trục này.",
        widths=[2.6, 2.4, 2.2, 1.8, 7.6], wide=True)
    rp.p(
        "Hàm `degrade()` mô phỏng sáu trục đó theo thứ tự vật lý của một trang scan: tay viết → mực "
        "loang trên giấy → mất phân giải khi chụp → nhoè quang học → giấy ngà và tương phản kém → "
        "nhiễu cảm biến. Mọi phép biến đổi đều là nhiễu ảnh, **không đụng tới cấu trúc bộ thủ**, nên "
        "nhãn vẫn đúng. Kết quả: `hit@1` của tập render tụt từ 0,7018 xuống 0,1986 — chỉ còn dễ hơn "
        "scan thật 2,8 lần thay vì 9 lần. Phần 2,8 lần còn lại là khác biệt **cấu trúc** (bút lông, "
        "lối hành thư: nét nối liền, tỉ lệ bộ phận xê dịch, có nét bị lược) và không thể sinh ra từ "
        "một ảnh in bằng xử lý ảnh.")

    rp.h("3.4. Fine-tune", 2)
    rp.p(
        "Mục tiêu học là **bất biến kiểu chữ**: cùng một mã Unicode qua font khác nhau phải cho "
        "cùng một vector. Nhãn duy nhất là mã Unicode — cố ý **không** dùng `RADICAL` hay "
        "`STROKE_NUM`, vì huấn luyện trên một nhãn rồi báo cáo chính chỉ số của nhãn đó là đo lại "
        "tập huấn luyện chứ không phải đánh giá.")
    rp.table(
        ["Thành phần", "Lựa chọn"],
        [["Backbone", "Chinese-CLIP ViT-B/16, giữ nguyên `visual_projection` (512 chiều)"],
         ["Hàm mất mát", "SupCon (InfoNCE có giám sát), nhiệt độ 0,07"],
         ["Lấy mẫu", "**P×K: 24 lớp × 2 view mỗi batch**"],
         ["Dữ liệu", "141.981 ảnh / 23.440 lớp (đã trừ held-out)"],
         ["Suy giảm ảnh", "`degrade()` chạy **trực tuyến** trong DataLoader, p = 0,6, strength = 1,0"],
         ["Tối ưu", "AdamW, lr 1e−5, warmup 300 bước, cosine, bf16, gradient checkpointing"],
         ["Chi phí", "**3,7 GB VRAM**, ~4 phút/epoch, 6 epoch (~30 phút trên một H100 dùng chung)"]],
        caption="Bảng 3. Cấu hình fine-tune.",
        widths=[3.6, 13.0])
    rp.p(
        "Hai lựa chọn trong bảng trên là **điều kiện sống còn chứ không phải tinh chỉnh**, và cả "
        "hai đều đến từ một lần thất bại được ghi ở mục 6.3:")
    rp.bullet(
        "**Lấy mẫu P×K.** Một batch 48 ảnh rút ngẫu nhiên từ 23.440 lớp chỉ chứa khoảng 0,05 cặp "
        "dương theo kỳ vọng — gần như mọi batch sẽ không có gì để kéo lại gần nhau, và hàm mất mát "
        "trở thành hằng số. P×K bảo đảm mỗi ảnh luôn có đúng K−1 ảnh cùng lớp.")
    rp.bullet(
        "**SupCon thay vì softmax có biên góc (ArcFace).** SupCon không có tham số học được nào "
        "trong hàm mất mát, nên không có một head khởi tạo ngẫu nhiên bơm nhiễu ngược về backbone.")
    rp.p(
        "Suy giảm ảnh chạy **trực tuyến lúc nạp dữ liệu** chứ không render sẵn ra đĩa: bộ sinh số "
        "ngẫu nhiên gieo theo bộ ba (seed, epoch, chỉ số ảnh) nên mỗi epoch mỗi ảnh có một biến thể "
        "khác, vẫn tái lập được, và không tốn thêm dung lượng nào.")

    # ---------------------------------------------------------------- 4
    rp.h("4. Phương pháp đánh giá")
    rp.p(
        "Phần lớn công sức của đồ án nằm ở chỗ này, vì một chỉ số sai làm mọi kết luận phía sau vô "
        "nghĩa. Ba tầng đánh giá, xếp theo độ tin cậy tăng dần:")
    rp.bullet(
        "**Chỉ số proxy** trên toàn corpus: `radical@k` (tỉ lệ láng giềng cùng bộ thủ), "
        "`stroke_mae@k`, `ids_jaccard@k`. Rẻ, chạy được trên 26.044 ký tự, nhưng là **nhãn yếu** — "
        "mục 6.2 cho thấy chúng có trần riêng.")
    rp.bullet(
        "**Nhãn thật từ tên file** trên 59 ảnh scan: giải mã Telex ra chữ Quốc Ngữ, tra từ điển ra "
        "tập ký tự đúng. `hit@k` = tỉ lệ ảnh có ít nhất một ký tự đúng trong top-k. Mức ngẫu nhiên "
        "khoảng 4,7·10⁻⁴, nên đây là bài kiểm tra khó và trung thực.")
    rp.bullet(
        "**Nhãn do người gán** cho Phần 2: với mỗi ảnh, người đọc chọn đúng một ứng viên trong "
        "nhóm cùng âm. Đã gán đủ 59/59 ảnh.")
    rp.p(
        "Cột `confidence` trong bảng nhãn là **đánh giá của người đọc ảnh, không suy ra từ bất kỳ "
        "điểm số nào của mô hình.** Điều này là cố ý: phép kiểm độ nhạy dùng chính cột này để lọc ra "
        "tập nhãn chắc chắn rồi chấm lại mô hình trên đó, nên nếu `confidence` lấy từ điểm "
        "similarity thì phép kiểm ấy thành vòng tròn — lấy mô hình chấm mô hình. Thang gồm ba mức: "
        "**high** (nét quyết định đọc được, mọi ứng viên còn lại khác ở bộ thủ hoặc số nét thấy rõ), "
        "**med** (còn ít nhất một biến thể gần trùng chỉ khác ở chi tiết đã mờ trong bản scan), "
        "**low** (nét quyết định không đọc được, lựa chọn dựa vào văn cảnh nhiều hơn hình). Phân bố "
        "cuối cùng là 53 high / 4 med / 2 low.")
    rp.p(
        "Các ảnh scan là những dòng mở đầu *Truyện Kiều*, nên nhãn được gán bằng so hình trước rồi "
        "đối chiếu với chính tả Nôm chuẩn của tác phẩm. **Khi hình và văn cảnh mâu thuẫn thì hình "
        "thắng**, và `confidence` bị hạ xuống `med` — bản scan viết \"qua\" là 〔戈〕 chứ không phải "
        "〔過〕, \"năm\" là 〔𢆥〕 chứ không phải 〔年〕.")
    rp.p(
        "**Mọi tỉ lệ đều kèm khoảng tin cậy nhị thức 95%.** Với n = 59 và p ≈ 0,5, khoảng tin cậy "
        "rộng khoảng ±13 điểm, nên bất kỳ cải thiện nào dưới ~15 điểm đều không chứng minh được. "
        "Đây là giới hạn cứng của tập test hiện có, và báo cáo tuân thủ nó một cách nhất quán: "
        "trong toàn bộ đồ án chỉ có đúng một so sánh vượt qua được ngưỡng này (mục 5.4).")

    # ---------------------------------------------------------------- 5
    rp.h("5. Kết quả")

    rp.h("5.1. So sánh backend trên toàn corpus", 2)
    rp.table(
        ["Backend", "dtype", "chiều", "radical@k ↑", "stroke_mae@k ↓", "ids_jaccard@k ↑", "img/s (H100)"],
        [["`resnet18` (bản gốc)", "fp32", "512", "0,2316", "2,8942", "0,1031", "3950"],
         ["`resnet18-gray`", "fp32", "512", "0,2202", "2,5658", "0,1020", "3937"],
         ["`chinese-clip` (B/16)", "fp32", "512", "0,4843", "2,6296", "0,1997", "358"],
         ["**`chinese-clip-large`**", "fp16", "768", "**0,5361**", "2,6992", "0,2113", "354"],
         ["`chinese-clip-huge`", "fp16", "1024", "0,5147", "2,6547", "**0,2144**", "283"],
         ["`dinov2`", "fp32", "1536", "0,3081", "**2,3397**", "0,1251", "306"]],
        caption="Bảng 4. Toàn bộ corpus 26.044 glyph, top-K = 20. Đậm là tốt nhất mỗi cột.",
        widths=[3.9, 1.5, 1.5, 2.2, 2.6, 2.6, 2.3], wide=True)
    rp.p(
        "Chinese-CLIP thắng với biên độ lớn: `radical@k` gấp **2,31 lần** bản gốc và `ids_jaccard` "
        "gấp 2,05 lần, mà gần như không tốn thêm thời gian (354 so với 358 ảnh/giây). Giả thuyết "
        "ban đầu — một prior quen với glyph Hán sẽ thắng một CNN ImageNet — được xác nhận.")
    rp.p(
        "DINOv2 thua ở bộ thủ nhưng **thắng ở số nét**, và chỉ trùng 12,9% kết quả top-k với họ "
        "CLIP. Nó không đơn thuần là kém hơn mà mạnh ở một trục khác, nên là ứng viên tốt cho "
        "ensemble.")

    rp.h("5.2. Trần của chính chỉ số proxy", 2)
    rp.p(
        "Một câu hỏi được đặt ra: `radical@k` ≈ 0,5 có phải đã bão hoà? Nhóm kiểm hai giả thuyết.")
    rp.p(
        "Giả thuyết \"tập top-k thiếu ứng viên đúng\" **bị bác bỏ**: trần lý thuyết của `radical@20` "
        "là 0,9862, tức chỉ 3% query bị chạm trần; và `radical@1` cũng chỉ đạt 0,6331, nên chỗ "
        "nghẽn nằm ngay ở láng giềng gần nhất chứ không ở đuôi danh sách.")
    rp.p(
        "Nhưng phép kiểm làm lộ ra một điều khác, và quan trọng hơn: **`RADICAL` không phải nhãn "
        "thị giác thuần tuý.** Bộ thủ chỉ thực sự xuất hiện trong hình chữ ở 56,7% số ký tự — phần "
        "còn lại mang bộ thủ vì lý do từ nguyên hoặc quy ước tra cứu, không phải vì trông giống. "
        "Một oracle *biết trước* phân rã IDS cũng chỉ đạt `radical@20` = 0,7338. So với trần thực tế "
        "đó, `chinese-clip-large` (0,5361) đang ở **73% quãng đường** và **24,4 lần** mức ngẫu "
        "nhiên. Kết luận: chưa bão hoà, nhưng cũng không nên tiếp tục tối ưu cho một chỉ số mà bản "
        "thân nó chỉ đúng ở hai phần ba.")

    rp.h("5.3. Có nên thay tìm kiếm chính xác bằng chỉ mục xấp xỉ?", 2)
    rp.p(
        "Đã đo trên dữ liệu thật: HNSW đạt recall 99,1% so với tìm chính xác ở N = 26.044. Nhưng ở "
        "cỡ corpus hiện tại, **tìm kiếm chỉ chiếm 6% tổng thời gian** — 94% còn lại là trích đặc "
        "trưng, thứ mà ANN không đụng tới. Điểm hoà vốn nằm quanh N ≈ 300.000. Kết luận: giữ "
        "`IndexFlatIP`. Nguyên lý về ANN là đúng, nhưng chưa tới lúc dùng.")

    rp.h("5.4. Kết quả trên ảnh scan thật — phép đo chính", 2)
    rp.p(
        "Đây là phép đo duy nhất không phải proxy, và là con số nên dùng để đánh giá đồ án. Không "
        "một pixel nào của `test_images/` đi vào huấn luyện; điều này được bảo đảm bằng một "
        "`assert` chạy trước mỗi lần huấn luyện chứ không bằng quy ước.")
    rp.table(
        ["Mô hình", "hit@1", "hit@10", "hit@20", "MRR", "KTC 95% của hit@1"],
        [["`chinese-clip` zero-shot (B/16)", "0,0169", "0,0678", "—", "0,0327", "—"],
         ["`chinese-clip-large` zero-shot", "0,0847", "0,1864", "0,2034", "0,1143", "[0,028; 0,187]"],
         ["**`chinese-clip-ft`** (fine-tune)", "**0,5932**", "**0,7966**", "**0,8644**", "**0,6691**",
          "**[0,457; 0,719]**"]],
        caption="Bảng 5. Phần 1 — tìm trên toàn bộ 26.044 ký tự, n = 59 ảnh scan thật.",
        widths=[5.2, 1.9, 1.9, 1.9, 1.9, 3.8], wide=True)
    rp.p(
        "**Hai khoảng tin cậy rời hẳn nhau.** Đây là so sánh duy nhất trong đồ án mà n = 59 đỡ nổi "
        "một kết luận — và không phải vì mẫu lớn hơn, mà vì hiệu ứng đủ lớn: 35/59 so với 5/59, hơn "
        "mức ngẫu nhiên 1.250 lần.")
    rp.p(
        "Trước fine-tune, kết quả thấp **không phải vì mô hình kém mà vì lệch phân phối**. Triệu "
        "chứng là hiệu ứng hub: ký tự 〔攅〕 được trả về ở vị trí một cho sáu query khác nhau. Query "
        "là chữ viết tay lối hành thư, corpus là khải thư in bằng font; hai phân phối này khác nhau "
        "ở cả nét lẫn hình học. Nhóm đã thử chuẩn hoá hình học (cắt sát nét mực, đệm vào khung "
        "vuông) và kết quả nằm trong khoảng nhiễu — nghĩa là phần lệch hình học có thật nhưng không "
        "phải nguyên nhân chính. Fine-tune mới là thứ giải quyết được.")
    rp.p("Phần 2, trên cùng bộ nhãn do người gán:")
    rp.table(
        ["Phương pháp", "Đúng top-1", "MRR", "KTC 95%"],
        [["Chọn ngẫu nhiên trong nhóm", "0,1987", "—", "—"],
         ["`histogram` (bản gốc)", "26/59 = 0,4407", "0,6070", "[0,312; 0,576]"],
         ["`embedding` zero-shot (large)", "28/59 = 0,4746", "0,6566", "[0,343; 0,609]"],
         ["**`embedding` fine-tune**", "**48/59 = 0,8136**", "**0,8983**", "**[0,691; 0,903]**"]],
        caption="Bảng 6. Phần 2 — xếp hạng trong nhóm cùng âm Quốc Ngữ, n = 59.",
        widths=[5.6, 3.6, 2.4, 3.4], wide=True)
    rp.p(
        "Phép thử độ nhạy trên 53 nhãn `high` cho 45/53 = 0,8491 so với histogram 0,4717, nên kết "
        "luận không phụ thuộc vào sáu nhãn `med`/`low` còn lại.")
    rp.p(
        "Bảng 6 **sửa lại một kết luận trước đó của chính nhóm**. Với mô hình zero-shot, khoảng cách "
        "giữa embedding và histogram chỉ là *hai ảnh trên 59* — nằm gọn trong nhiễu — nên khuyến "
        "nghị dùng embedding khi ấy phải dựa vào chất lượng điểm số chứ không dám viện độ chính xác. "
        "Sau fine-tune, khoảng cách là 22 ảnh và hai khoảng tin cậy rời nhau: độ chính xác đã đủ để "
        "tự đứng làm lý do.")

    rp.h("5.5. Có phải mô hình chỉ ghi nhớ 23.440 danh tính?", 2)
    rp.p(
        "Đây là phản biện hiển nhiên nhất với một kết quả tăng mạnh như vậy, nên nhóm thiết kế phép "
        "kiểm cho nó ngay từ đầu. **2.604 mã Unicode bị gỡ toàn bộ view khỏi tập huấn luyện** — chia "
        "theo *lớp* chứ không theo ảnh, vì chia theo ảnh thì lớp nào cũng có mặt ở cả hai bên và "
        "phép đo mất sạch ý nghĩa. Query là một bản render font, gallery là ảnh corpus.")
    rp.table(
        ["", "Lớp đã thấy", "Lớp chưa thấy"],
        [["Trước huấn luyện", "0,7938", "0,8041"],
         ["Sau huấn luyện", "0,9954", "**0,9939**"]],
        caption="Bảng 7. hit@1 render→corpus. Hai mẫu cùng cỡ 2.604 lớp và cùng cách chọn ngẫu nhiên.",
        widths=[5.0, 4.0, 4.0])
    rp.p(
        "**Bằng nhau.** Không có khoảng cách ghi nhớ. Điều này khớp với cơ chế: SupCon không có tâm "
        "lớp học được nên không có gì để nhớ — thứ nó học là *phép so hai ảnh*, và phép đó chuyển "
        "sang ký tự chưa từng gặp nguyên vẹn. Cần nói thêm rằng hai mẫu phải được chọn ngẫu nhiên "
        "cùng cách: nếu lấy \"lớp đã thấy\" theo thứ tự mã Unicode thì mẫu sẽ thiên về khối CJK cơ "
        "bản (chữ phổ thông, tự dạng quen) trong khi held-out rải đều cả Ext-B, và hai con số không "
        "còn so được với nhau.")

    # ---------------------------------------------------------------- 6
    rp.h("6. Kết quả âm")
    rp.p(
        "Bốn kết quả dưới đây đều là thất bại hoặc bác bỏ, và đều được giữ lại vì chúng loại trừ "
        "được những hướng mà người đọc báo cáo này rất dễ thử lại.")

    rp.h("6.1. Giả thuyết \"lỗi nằm ở conv1\" — sai", 2)
    rp.p(
        "Bản gốc thay `conv1` bằng lớp 1 kênh khởi tạo ngẫu nhiên, tức vứt bỏ bộ lọc tầng đầu đã "
        "pretrain. Giả thuyết tự nhiên là chỉ cần vá dòng đó. Nhóm dựng `resnet18-gray` gieo `conv1` "
        "bằng tổng các bộ lọc RGB pretrain — kết quả `radical@k` **giảm nhẹ** (0,2202 so với "
        "0,2316). Sửa một dòng không cứu được kiến trúc; đây chính là bằng chứng biện minh cho việc "
        "đổi hẳn sang ViT thay vì vá baseline.")

    rp.h("6.2. Scale có lợi, rồi đảo chiều", 2)
    rp.p(
        "base → large là +10,7% tương đối trên `radical@k`, nhưng large → huge là **−4,0%** trong "
        "khi tốn thêm 33% số chiều và 20% thời gian. Đường cong đã quay đầu. Đây là chữ ký của "
        "\"thêm dung lượng cho một mục tiêu huấn luyện sai domain\" — và cũng chính là lý do khiến "
        "fine-tune trở thành hướng đáng làm nhất, thay vì đổi sang checkpoint lớn hơn.")

    rp.h("6.3. ArcFace gây sụp đổ biểu diễn", 2)
    rp.p(
        "Thiết kế fine-tune đầu tiên dùng ArcFace trên 23.440 lớp (scale 32, biên 0,30). Sau một "
        "epoch:")
    rp.table(
        ["Chỉ số", "Trước", "Sau 1 epoch"],
        [["render→corpus, lớp đã thấy", "0,7938", "**0,0035**"],
         ["render→corpus, lớp chưa thấy", "0,8041", "**0,0050**"],
         ["scan thật hit@1", "0,0169", "**0,0000**"],
         ["hàm mất mát", "20,5", "19,4 (gần như đứng yên)"]],
        caption="Bảng 8. Diễn biến khi huấn luyện bằng ArcFace.",
        widths=[6.4, 3.3, 4.3])
    rp.p(
        "Sụp ở **cả hai** phía nên đây không phải overfit mà là **sụp đổ biểu diễn**: mọi ảnh dồn về "
        "một vector. Nguyên nhân nằm ở hình dạng dữ liệu chứ không phải siêu tham số — 23.440 lớp mà "
        "mỗi lớp chỉ khoảng 6 ảnh. Head ArcFace khởi tạo ngẫu nhiên với 23.440 tâm không bao giờ kịp "
        "tổ chức, nên thứ nó truyền ngược về backbone gần như là nhiễu. Và vì Adam chuẩn hoá theo độ "
        "lớn gradient, \"tốc độ học nhỏ\" 1e−5 vẫn dịch chuyển trọng số đủ để phá cấu trúc pretrain "
        "trong khoảng 3.000 bước; cắt chuẩn gradient không cứu được, vì nhiễu bị cắt chuẩn vẫn là "
        "nhiễu. Đây là lý do mục 3.4 chọn SupCon.")
    rp.p(
        "Ba công cụ chẩn đoán được thêm vào sau lần này, và nên có trong bất kỳ thí nghiệm tương tự "
        "nào: (a) đánh giá **giữa epoch** để sự cố lộ ra sau một phút thay vì sau một giờ; (b) một "
        "cột đo **cosine trung bình giữa 2.000 vector corpus ngẫu nhiên** — nó chạy về 1,0 khi sụp "
        "đổ, và nếu không có nó thì \"học hỏng\" với \"sụp đổ\" trông giống hệt nhau trên `hit@1` "
        "trong khi hai thứ đó phải sửa theo hai cách khác nhau; (c) biết trước chữ ký sụp đổ của "
        "SupCon là hàm mất mát đứng đúng ở ln(P·K − 1). Trong lần chạy thành công, cột cosine đi từ "
        "**0,890 xuống 0,000** — tức không gian giãn ra tới mức gần trực giao. Con số 0,890 lúc "
        "zero-shot tự nó giải thích vì sao khả năng phân biệt trước đây yếu.")

    rp.h("6.4. Một nhãn sai đã tự làm giảm kết quả của chính nhóm", 2)
    rp.p(
        "Khi soát lại 11 dòng `med`/`low`, nhóm phát hiện một nhãn gán sai: bộ bên trái của chữ là "
        "〔土〕 (nét dọc dừng ở vạch ngang) chứ không phải 〔扌〕 (móc kéo xuống dưới vạch). Sửa lại "
        "làm một ảnh chuyển từ cột embedding sang cột histogram, khiến khoảng cách giữa hai phương "
        "pháp **co từ bốn ảnh xuống hai** — tức đi ngược lợi ích của kết luận mà nhóm đang muốn "
        "chứng minh. Ghi lại điều này vì nó là thước đo trực tiếp cho việc một bảng số ở n = 59 "
        "nhạy tới mức nào với đúng một nhãn.")

    # ---------------------------------------------------------------- 7 rerank
    rp.h("7. Xếp hạng lại bằng siêu dữ liệu — một kết quả âm")
    rp.p(
        "Ba cột siêu dữ liệu ngôn ngữ học (`RADICAL`, `STROKE_NUM`, `SHAPE_MORPH`) mô tả những "
        "thuộc tính mà mô hình ảnh không nhìn thấy. Câu hỏi tự nhiên: dùng chúng để **xếp hạng lại** "
        "danh sách ứng viên mà tầng ảnh trả về thì có tốt hơn không? Chúng tôi cài đặt tầng đó và "
        "trả lời bằng số. Câu trả lời là **chưa**, và lý do cụ thể hơn nhiều so với \"không hiệu "
        "quả\".")

    rp.h("7.1. Vì sao không thể chấm bằng chỉ số cũ", 2)
    rp.p(
        "Ba chỉ số proxy ở mục 4 được suy ra từ **đúng ba cột** mà tầng xếp hạng lại dùng làm đặc "
        "trưng. Chấm nó bằng `radical@k` là vòng tròn: điểm số sẽ leo về trần 0,978 mà không chứng "
        "minh được gì, vì mô hình được đưa sẵn đáp án. Vì vậy phép đo trung thực phải dùng nhãn "
        "**độc lập** với ba cột đó — ở đây là mã Unicode của chính ký tự, trên tập ảnh render "
        "(n = 600, khoảng tin cậy khoảng ±3,5 điểm, đủ hẹp để phân giải mức cải thiện đáng quan tâm).")

    rp.h("7.2. Bốn cấu hình, và điều mỗi cấu hình trả lời", 2)
    rp.table(
        ["Cấu hình", "hit@1", "KTC 95%", "MRR"],
        [["Tầng 1, không xếp hạng lại", "0,7467", "[0,710; 0,780]", "0,8193"],
         ["Xếp hạng lại, **oracle** (cố tình gian lận)", "**0,9217**", "[0,897; 0,941]", "0,9446"],
         ["Xếp hạng lại, thật", "0,7233", "[0,686; 0,758]", "0,8052"],
         ["Xếp hạng lại, **không chuẩn hoá tín hiệu**", "0,1883", "[0,159; 0,222]", "0,2447"]],
        caption="Bảng 9. Xếp hạng lại trên 600 ảnh render, tầng 1 là chinese-clip-large. Cấu hình "
                "oracle đọc trộm siêu dữ liệu thật của truy vấn — nó là chẩn đoán, không phải kết quả.",
        widths=[6.4, 2.6, 3.4, 2.4], wide=True)
    rp.p("Ba điều rút ra, theo thứ tự quan trọng:")
    rp.bullet(
        "**Chuẩn hoá tín hiệu trước khi trộn là bắt buộc.** Trộn điểm thô làm mất 56 điểm hit@1 "
        "(0,7467 → 0,1883). Lý do: với trọng số mặc định, một ứng viên phải thắng về độ giống ảnh "
        "hơn 0,364 cosine mới bù nổi một lần lệch bộ thủ, mà khoảng cách cosine trong top-20 không "
        "bao giờ rộng đến thế. Trộn thô vì vậy biến phép xếp hạng thành **sắp xếp từ điển** — bộ "
        "thủ trước, rồi số nét, rồi hình dạng — đẩy mô hình ảnh xuống làm tiêu chí phá hoà. Đây là "
        "một lỗi thật, và nó chỉ lộ ra khi có phép đo không vòng tròn.")
    rp.bullet(
        "**Siêu dữ liệu có trần cao.** Cấu hình oracle được +17,5 điểm. Nghĩa là nếu biết đúng bộ "
        "thủ, số nét và phân rã hình dạng của ảnh truy vấn thì ý tưởng này thắng rõ rệt. Vấn đề "
        "không nằm ở ý tưởng.")
    rp.bullet(
        "**Nhưng trần đó chưa với tới được.** Cấu hình thật đạt 0,7233, tức **hơi thấp hơn** mốc "
        "không xếp hạng lại. Toàn bộ khoảng cách 17,5 điểm giữa oracle và thực tế nằm ở một bước "
        "duy nhất: suy ra siêu dữ liệu của ảnh truy vấn.")

    rp.h("7.3. Chỗ nghẽn, định lượng", 2)
    rp.p(
        "Ảnh truy vấn là một bản scan chưa biết là chữ gì, nên không tra được siêu dữ liệu của nó. "
        "Cách thay thế là **bỏ phiếu từ láng giềng**: lấy top-50 ký tự giống nhất trên toàn corpus "
        "rồi lấy bộ thủ chiếm đa số. Đo trực tiếp độ chính xác của bước này:")
    rp.table(
        ["Cách đoán bộ thủ của ảnh truy vấn", "Độ chính xác"],
        [["Đoán đều tay giữa 214 bộ", "0,0047"],
         # Ghi kèm âm đọc: 〔口〕 vốn là một hình vuông, không thêm chú thì người đọc tưởng font hỏng.
         ["Luôn đoán bộ phổ biến nhất, bộ 〔口〕 (khẩu)", "0,0489"],
         ["Bỏ phiếu từ láng giềng, tầng 1 = `chinese-clip-large`", "0,1186"],
         ["Bỏ phiếu từ láng giềng, tầng 1 = **`chinese-clip-ft`**", "**0,6610**"]],
        caption="Bảng 10. Độ chính xác suy ra bộ thủ của ảnh truy vấn, đo trên 59 scan thật.",
        widths=[9.4, 3.2])
    rp.p(
        "Với tầng 1 zero-shot, phép bỏ phiếu chỉ đúng 11,86% — hơn đoán mò 2,4 lần, nhưng **sai "
        "88% số lần**, trong khi bộ thủ chiếm trọng số 0,20 trong điểm tổng. Tầng xếp hạng lại vì "
        "vậy đang được nuôi bằng nhiễu.")
    rp.p(
        "Nguyên nhân có tính cơ chế chứ không phải lỗi cài đặt: phép bỏ phiếu lấy từ láng giềng của "
        "tầng 1, mà tầng 1 zero-shot chỉ đúng 8,47% trên scan thật (Bảng 5), nên đa số phiếu là của "
        "chữ sai. **Thay tầng 1 bằng mô hình đã fine-tune thì độ chính xác nhảy lên 0,6610 — gấp "
        "5,6 lần.** Mắt xích yếu không nằm ở thiết kế của phép suy luận mà ở chất lượng của tầng "
        "đứng trước nó.")
    rp.p(
        "Trên 59 scan thật, xếp hạng lại đưa Phần 2 từ 48/59 lên 50/59 với tầng 1 fine-tune. Đúng "
        "hai ảnh — nằm gọn trong khoảng nhiễu ở n = 59, nên **không được đọc là cải thiện**. Cơ chế "
        "đã chạy đúng, nhưng tập test hiện có không đủ để chứng minh.")

    rp.h("7.4. Một cảnh báo về phép đo", 2)
    rp.p(
        "Tập ảnh render **không** dùng để đánh giá `chinese-clip-ft` được, vì chính chúng là dữ "
        "liệu huấn luyện của mô hình đó. Chúng tôi đã thử cứu bằng cách chỉ lấy 2.604 lớp Unicode "
        "held-out, rồi thử tiếp trên bản đã suy giảm — cả hai đều bão hoà trên 0,98. Nghĩa là phép "
        "đo không vòng tròn ở mục 7.1 chỉ dùng được cho mô hình zero-shot; với mô hình đã fine-tune "
        "thì chỉ còn 59 ảnh scan thật, và ta quay lại đúng nút thắt cỡ mẫu ở mục 8.")

    # ---------------------------------------------------------------- 8
    rp.h("8. Hạn chế")
    rp.bullet(
        "**n = 59 là toàn bộ dữ liệu scan hiện có**, và là nút thắt lớn nhất. Mọi so sánh trong đồ "
        "án trừ mục 5.4 đều nằm trong khoảng nhiễu. Cần n ≈ 150 để khoảng tin cậy xuống ±8 điểm.")
    rp.bullet(
        "**Mô hình fine-tune chỉ được đo trên một lần chạy, một seed.** Chênh lệch giữa các epoch "
        "(0,51–0,59) đã bằng cỡ nhiễu của phép đo, nên 0,5932 nên đọc là \"quanh 0,55\".")
    rp.bullet(
        "**Backbone fine-tune là ViT-B/16 chứ không phải large**, vì large không vừa VRAM lúc huấn "
        "luyện trên máy dùng chung. Chưa biết fine-tune large sẽ cho kết quả bao nhiêu.")
    rp.bullet(
        "Nhãn Phần 2 do **so hình** mà ra, không phải do chuyên gia Hán-Nôm đọc. Còn 6/59 dòng ở "
        "mức `med`/`low`.")
    rp.bullet(
        "**Tầng xếp hạng lại chưa được kiểm chứng ở cấu hình tốt nhất.** Bảng 9 đo với tầng 1 "
        "zero-shot; khi tầng 1 là mô hình fine-tune thì bước suy ra bộ thủ tốt lên 5,6 lần, nhưng "
        "chưa ai chạy lại đủ bốn cấu hình trên tầng 1 đó. Kết luận \"chưa dùng được\" ở mục 7 vì "
        "thế chỉ đúng cho cấu hình đã đo.")
    rp.bullet(
        "Ảnh query là chữ viết tay lối hành thư còn corpus là khải thư in — mọi con số đo trên đúng "
        "một cặp phân phối này, chưa nói được gì về các lối viết khác.")
    rp.bullet(
        "Recall của ANN chỉ đo ở N = 26.044; recall của HNSW thường giảm khi N tăng, nên 99,1% "
        "không được suy diễn cho mốc một triệu.")

    # ---------------------------------------------------------------- 8
    rp.h("9. Kết luận và hướng tiếp theo")
    rp.bullet(
        "**Chuyển sang Chinese-CLIP.** Hơn baseline ResNet18 2,31 lần ở `radical@k` với chi phí "
        "gần như bằng không. Không scale tiếp lên ViT-H: đường cong đã quay đầu.", numbered=True)
    rp.bullet(
        "**Fine-tune là đòn bẩy lớn nhất, và đã đo được.** `hit@1` trên scan thật đi từ 0,0847 lên "
        "**0,5932**, Phần 2 từ 0,4746 lên **0,8136**, với bằng chứng rằng phần cải thiện không đến "
        "từ ghi nhớ.", numbered=True)
    rp.bullet(
        "**Giữ tìm kiếm chính xác** `IndexFlatIP` cho tới khi corpus vượt khoảng 300.000 ký tự.",
        numbered=True)
    rp.bullet(
        "**Xếp hạng lại bằng siêu dữ liệu chưa dùng được, nhưng biết rõ vì sao.** Oracle cho thấy "
        "còn +17,5 điểm dư địa; chỗ nghẽn là bước suy ra bộ thủ của ảnh truy vấn, hiện đúng 11,9% "
        "với tầng 1 zero-shot và 66,1% với tầng 1 fine-tune. Đo lại toàn bộ trên tầng 1 fine-tune "
        "là việc rẻ nhất còn chưa làm (mục 7).", numbered=True)
    rp.bullet(
        "**Việc tiếp theo, theo thứ tự chi phí.** (a) *Hard negative mining*: hàm mất mát hiện rơi "
        "về 0,0068 chỉ sau 400 bước vì 24 lớp lấy ngẫu nhiên từ 23.440 thì phân biệt quá dễ — phải "
        "nhồi vào cùng batch những ký tự trông giống nhau thì mới còn tín hiệu để học. (b) Scan "
        "thêm ảnh thật để gỡ nút thắt n = 59. (c) Bổ sung dữ liệu thư pháp bút lông thật (MCCD, "
        "NomNaOCR) — thứ mà render font về nguyên tắc không sinh ra được.", numbered=True)

    # ---------------------------------------------------------------- Tham khảo
    rp.h("Tài liệu tham khảo")
    for i, ref in enumerate(REFERENCES, 1):
        p = rp.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.first_line_indent = Cm(-0.7)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        rich(p, f"[{i}] {ref}", size=8.5)

    # ---------------------------------------------------------------- Phụ lục
    # Phụ lục về 1 cột: các dòng lệnh dài không xuống dòng đẹp trong cột rộng 8 cm.
    rp.columns(1)
    rp.pagebreak()
    rp.h("Phụ lục A. Tái lập toàn bộ số liệu")
    rp.p(
        "Mọi con số trong báo cáo sinh ra từ các lệnh dưới đây, chạy tuần tự **từ trong thư mục dự "
        "án**. Mọi script đều phân giải `./images`, `./output`, `./test_images` và các file `.xlsx` "
        "theo thư mục làm việc hiện tại, nên chạy từ chỗ khác sẽ hỏng.")
    rp.p("**Chuẩn bị môi trường.** Cần Python 3.10+ và một GPU có ít nhất 6 GB trống.")
    rp.code([
        "# nếu máy đã có sẵn torch hệ thống, thêm --system-site-packages",
        "python -m venv .venv",
        ".venv/bin/pip install -r requirements.txt",
        "",
        "# đặt hai file .xlsx vào thư mục gốc, giải nén images.zip -> ./images",
        "bash download_fonts.sh                        # tải 7 font Hán-Nôm vào ./fonts",
    ])
    rp.p("**Mục 5.1 — so sánh backend.** Mỗi lệnh cache embedding, chạy lại sẽ dùng cache.")
    rp.code([
        ".venv/bin/python search_all_chars_in_corpus.py --backend resnet18          --device cuda",
        ".venv/bin/python search_all_chars_in_corpus.py --backend resnet18-gray     --device cuda",
        ".venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip      --device cuda",
        ".venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip-large \\",
        "        --device cuda --dtype fp16",
        ".venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip-huge \\",
        "        --device cuda --dtype fp16",
        ".venv/bin/python search_all_chars_in_corpus.py --backend dinov2            --device cuda",
        ".venv/bin/python benchmark_extractors.py      # -> benchmark_summary.xlsx (Bảng 4)",
    ])
    rp.p("**Mục 5.2 và 5.3 — trần của chỉ số proxy, và so sánh tìm kiếm chính xác với ANN.**")
    rp.code([
        ".venv/bin/python analyze_metric_ceiling.py    # trần lý thuyết + oracle IDS",
        ".venv/bin/python benchmark_search.py          # -> output/benchmark_search.xlsx",
    ])
    rp.p("**Mục 3.3 — sinh dữ liệu huấn luyện.** Bước này mất khoảng 20 phút trên 16 nhân CPU.")
    rp.code([
        ".venv/bin/python render_fonts.py              # 131.604 ảnh -> output/rendered/",
        ".venv/bin/python scan_augment.py --calibrate  # Bảng 2",
        ".venv/bin/python scan_augment.py --preview    # bảng ảnh để nhìn bằng mắt",
        "",
        "# KHÔNG cần --augment cho việc huấn luyện: finetune_glyph.py suy giảm ảnh trực tuyến.",
        "# Cờ này chỉ dùng khi muốn xem trước tập suy giảm trên đĩa.",
    ])
    rp.p("**Mục 3.4 — fine-tune.** Khoảng 30 phút, 3,7 GB VRAM.")
    rp.code([
        ".venv/bin/python finetune_glyph.py --smoke   # 200 lớp, kiểm đường ống trước",
        ".venv/bin/python finetune_glyph.py --epochs 6 --device cuda",
        "                                             # -> output/finetune/best.pt",
    ])
    rp.p(
        "**Mục 4, 5.4 — đánh giá trên scan thật.** Bỏ `--backend chinese-clip-ft` để lấy lại các "
        "con số zero-shot ở cùng bảng.")
    rp.code([
        ".venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip-ft \\",
        "        --dtype fp16 --device cuda",
        "",
        ".venv/bin/python evaluate_test_images.py --backend chinese-clip-ft --dtype fp16 \\",
        "        --device cuda --labels output/label_sheets/label_template.csv",
        "",
        "# phép thử độ nhạy: chỉ chấm trên 53 nhãn confidence=high",
        ".venv/bin/python evaluate_test_images.py --backend chinese-clip-ft --dtype fp16 \\",
        "        --device cuda --labels output/label_sheets/label_template.csv \\",
        "        --part 2 --min-confidence high",
        "",
        "# sinh lại phiếu gán nhãn (query + TOÀN BỘ ứng viên, kèm mã Unicode)",
        ".venv/bin/python evaluate_test_images.py --sheets",
    ])
    rp.p("**Mục 7 — xếp hạng lại bằng siêu dữ liệu.** Bốn cấu hình của Bảng 9, theo thứ tự.")
    rp.code([
        "# 55 unit test của tầng xếp hạng lại",
        ".venv/bin/python -m pytest -q",
        "",
        "S=\"--suite render --sample 600 --backend chinese-clip-large --dtype fp16\"",
        ".venv/bin/python evaluate_rerank.py $S --no-rerank      # tầng 1",
        ".venv/bin/python evaluate_rerank.py $S --oracle-meta    # oracle (chẩn đoán)",
        ".venv/bin/python evaluate_rerank.py $S                  # thật",
        ".venv/bin/python evaluate_rerank.py $S --normalize none # không chuẩn hoá",
        "",
        "# Bảng 10: độ chính xác suy ra bộ thủ, in kèm ở cuối suite scans",
        ".venv/bin/python evaluate_rerank.py --suite scans --backend chinese-clip-large",
        ".venv/bin/python evaluate_rerank.py --suite scans --backend chinese-clip-ft",
    ])
    rp.p(
        "Nhật ký huấn luyện đầy đủ, gồm cả lần ArcFace sụp đổ ở mục 6.3, nằm trong "
        "`output/finetune/train.log`. `BENCHMARK.md` giữ bản dài của mọi bảng phụ mà báo cáo này "
        "không chép lại.")

    rp.h("Phụ lục B. Nội dung gói nộp")
    rp.table(
        ["Đường dẫn", "Nội dung"],
        [["`src/`", "Toàn bộ mã nguồn do nhóm viết (xem bảng dưới)."],
         ["`data/test_images/`", "59 ảnh scan thật — tập đánh giá."],
         ["`data/label_template.csv`", "**Nhãn do người gán**, kèm cột `confidence`. Không sinh lại được bằng cách chạy code."],
         ["`data/DATA.md`", "Nguồn của từng tập dữ liệu và cách lấy phần không kèm theo."],
         ["`model/`", "Trọng số fine-tune, hoặc liên kết tải nếu vượt hạn mức nộp."],
         ["`report/`", "Báo cáo `.docx` và `.pdf`, cùng script sinh ra chúng."],
         ["`BENCHMARK.md`", "Bản dài: mọi bảng, mọi lần chạy, mọi kết quả âm."],
         ["`README.md`", "Hướng dẫn chạy nhanh."]],
        caption="Bảng 11. Cấu trúc gói nộp.",
        widths=[5.2, 11.4], wide=True)
    rp.table(
        ["Script", "Vai trò"],
        [["`feature_extractors.py`", "Sáu backend trích đặc trưng sau một giao diện chung."],
         ["`search_all_chars_in_corpus.py`", "Phần 1: nhúng toàn corpus, cache, tìm top-k bằng Faiss."],
         ["`search_use_QuocNgu_mapping.py`", "Phần 2: xếp hạng trong nhóm cùng âm, histogram và embedding."],
         ["`benchmark_extractors.py`", "So sánh backend theo ba chỉ số proxy (Bảng 4)."],
         ["`analyze_metric_ceiling.py`", "Trần lý thuyết và trần oracle của `radical@k` (mục 5.2)."],
         ["`benchmark_search.py`", "So tìm kiếm chính xác với ANN (mục 5.3)."],
         ["`evaluate_test_images.py`", "Đánh giá trên scan thật, sinh phiếu gán nhãn (mục 4, 5.4)."],
         ["`render_fonts.py`", "Sinh cặp dương bằng render đa font (mục 3.3)."],
         ["`scan_augment.py`", "Mô hình suy giảm ảnh, đã hiệu chỉnh theo số đo (Bảng 2)."],
         ["`finetune_glyph.py`", "Fine-tune SupCon (mục 3.4), kèm nhánh ArcFace để tái lập mục 6.3."],
         ["`rerank.py`", "Tầng xếp hạng lại bằng siêu dữ liệu ngôn ngữ học (mục 7)."],
         ["`evaluate_rerank.py`", "Ba suite đánh giá tầng xếp hạng lại, tách rõ vòng tròn và trung thực."],
         ["`test_rerank.py`, `test_evaluate_rerank.py`", "55 unit test."],
         ["`build_report.py`", "Sinh chính báo cáo này."]],
        caption="Bảng 12. Mã nguồn do nhóm viết.",
        widths=[5.6, 11.0], wide=True)

    rp.h("Tài liệu và tài nguyên ngoài")
    rp.p(
        "Các mô hình pretrain **không** kèm trong gói nộp; chúng được tải tự động từ Hugging Face "
        "khi chạy lần đầu:")
    rp.bullet("Chinese-CLIP: `OFA-Sys/chinese-clip-vit-base-patch16`, `…-large-patch14`, `…-huge-patch14`")
    rp.bullet("DINOv2: `facebook/dinov2-base`")
    rp.bullet("ResNet18: trọng số ImageNet của torchvision")
    rp.p(
        "Font Hán-Nôm (BabelStone Han, HanaMin A/B, I.Ming, LXGW WenKai TC, Noto Sans/Serif TC, "
        "Zhuque Fangsong) tải bằng `download_fonts.sh`; tất cả đều là font tự do, nguồn ghi trong "
        "chính script đó.")


def to_pdf(docx_path, out_dir):
    """Chuyển .docx -> .pdf bằng LibreOffice headless.

    Máy này không có TeX/pandoc, còn LibreOffice thì có sẵn — nên đây là đường duy nhất ra PDF.
    LibreOffice trả về mã 0 kể cả khi không xuất được file, nên phải kiểm sự tồn tại của file đích
    chứ không tin returncode.
    """
    if not shutil.which("soffice"):
        return None
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, docx_path],
                   capture_output=True, timeout=300)
    pdf = os.path.join(out_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
    return pdf if os.path.exists(pdf) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="report")
    parser.add_argument("--name", default="BaoCao")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    copied = ensure_fonts()
    if copied:
        print(f"Đã cài font vào ~/.fonts: {', '.join(copied)}")

    os.makedirs(args.out_dir, exist_ok=True)
    report = Report()
    build(report)
    docx_path = os.path.join(args.out_dir, args.name + ".docx")
    report.save(docx_path)
    print(f"Đã ghi {docx_path}")

    if not args.no_pdf:
        pdf = to_pdf(docx_path, args.out_dir)
        print(f"Đã ghi {pdf}" if pdf else "Không xuất được PDF (thiếu LibreOffice?)")


if __name__ == "__main__":
    main()
