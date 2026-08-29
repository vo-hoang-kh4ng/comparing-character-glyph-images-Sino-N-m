#!/usr/bin/env bash
# Tải font CJK cho render_fonts.py. Máy dev không có font CJK nào (`fc-list :lang=zh` = 0),
# và font là file lớn (~200MB) nên KHÔNG commit vào repo — thư mục `fonts/` đã gitignore.
# Tất cả đều là font tự do, dùng được cho nghiên cứu.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts && cd fonts

get() { [ -f "$(basename "$1")" ] || curl -fsSL -O "$1"; }

get https://www.babelstone.co.uk/Fonts/Download/BabelStoneHan.ttf
get https://github.com/cjkvi/HanaMinAFDKO/releases/download/8.030/HanaMinA.otf
get https://github.com/cjkvi/HanaMinAFDKO/releases/download/8.030/HanaMinB.otf
get https://github.com/ichitenfont/I.Ming/releases/download/8.10/I.Ming-8.10.ttf
get https://github.com/lxgw/LxgwWenkaiTC/releases/download/v1.522/LXGWWenKaiTC-Regular.ttf
get https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/15_NotoSerifTC.zip
get https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/19_NotoSansTC.zip
get https://github.com/TrionesType/zhuque/releases/download/v0.212/ZhuqueFangsong-v0.212.zip

for z in *.zip; do [ -d "${z%.zip}" ] || unzip -oq "$z" -d "${z%.zip}"; done
echo "Xong. $(find . -iname '*.[ot]tf' | wc -l) file font trong $(pwd)"
