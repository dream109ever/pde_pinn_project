import os
import subprocess
import sys
import json

def run_cloc_json():
    """以 JSON 格式运行 cloc，获取完美的结构化数据（包含所有文件明细和路径）"""
    result = subprocess.run(
        [
            "cloc", ".",
            "--json",
            "--by-file",
            "--fullpath",
            "--not-match-d=docs/_build|docs/html",
            "--exclude-dir=__pycache__,venv,.vscode,.idea,build,dist,results,bag,node_modules,html",
            "--exclude-ext=pyc,pyo,log,tmp,svg,json,csv,spec,html,txt,pyi,mo,po"
        ],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("执行 cloc 失败，请确保已安装 cloc 并且在项目根目录下。")
        sys.exit(1)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("解析 cloc JSON 输出失败。")
        sys.exit(1)
def process_cloc_data(cloc_json):
    """根据 cloc JSON 中的真实文件路径，精确读取每个文件的字符数，并按语言汇总"""
    lang_stats = {}
    for path, info in cloc_json.items():
        if path in ["header", "SUM"]:
            continue
        lang = info.get("language")
        code = info.get("code", 0)
        comment = info.get("comment", 0)
        blank = info.get("blank", 0)
        if not lang:
            continue
        clean_path = path.strip()
        possible_paths = [
            clean_path,
            clean_path.lstrip("./").lstrip(".\\"),
            os.path.normpath(clean_path)
        ]
        char_count = 0
        real_path = None
        for p in possible_paths:
            if os.path.exists(p):
                real_path = p
                break
        if real_path:
            try:
                with open(real_path, 'r', encoding='utf-8', errors='ignore') as f:
                    char_count = len(f.read())
            except Exception:
                char_count = 0
        if lang not in lang_stats:
            lang_stats[lang] = {
                "lang": lang,
                "files": 0,
                "blank": 0,
                "comment": 0,
                "code": 0,
                "total": 0,
                "chars": 0
            }
        lang_stats[lang]["files"] += 1
        lang_stats[lang]["blank"] += blank
        lang_stats[lang]["comment"] += comment
        lang_stats[lang]["code"] += code
        lang_stats[lang]["total"] += (blank + comment + code)
        lang_stats[lang]["chars"] += char_count
    data = []
    for lang, stats in lang_stats.items():
        total = stats["total"]
        stats["ratio"] = (stats["code"] / total * 100) if total > 0 else 0
        data.append(stats)
    data.sort(key=lambda x: x["code"], reverse=True)
    return data
def get_display_width(text):
    """计算字符串在终端中的显示宽度（中文字符算 2 个宽度）"""
    width = 0
    for char in text:
        if ord(char) > 127:
            width += 2
        else:
            width += 1
    return width
def print_report(data):
    """打印完美对齐的综合报告"""
    headers = ["语言", "文件数", "总行数", "代码行", "注释行", "空行", "代码占比", "字符数", "字符占比"]
    col_widths = [18, 10, 12, 12, 12, 12, 12, 12, 12]
    def format_row(items, widths, align_right=False):
        row_str = ""
        for i, item in enumerate(items):
            text = str(item)
            w = widths[i]
            current_w = get_display_width(text)
            padding = max(0, w - current_w)  
            if i == 0 or not align_right:
                row_str += text + " " * padding
            else:
                row_str += " " * padding + text
            row_str += "  "
        return row_str
    print("-" * 128)
    print(format_row(headers, col_widths, align_right=True))
    print("-" * 128)
    total_files = 0
    total_lines = 0
    total_code = 0
    total_comment = 0
    total_blank = 0
    total_chars = 0
    for item in data:
        total_chars += item["chars"]
    for item in data:
        total_files += item["files"]
        total_lines += item["total"]
        total_code += item["code"]
        total_comment += item["comment"]
        total_blank += item["blank"]
        char_ratio = (item["chars"] / total_chars * 100) if total_chars > 0 else 0
        row_data = [
            str(item["lang"]),
            str(item["files"]),
            f"{item['total']:,}",
            f"{item['code']:,}",
            f"{item['comment']:,}",
            f"{item['blank']:,}",
            f"{item['ratio']:.2f}%",
            f"{item['chars']:,}",
            f"{char_ratio:.2f}%"
        ]
        print(format_row(row_data, col_widths, align_right=True))
    print("-" * 128)
    total_ratio = total_code / total_lines * 100 if total_lines > 0 else 0
    total_row = [
        "总计",
        str(total_files),
        f"{total_lines:,}",
        f"{total_code:,}",
        f"{total_comment:,}",
        f"{total_blank:,}",
        f"{total_ratio:.2f}%",
        f"{total_chars:,}",
        f"100.00%"
    ]
    print(format_row(total_row, col_widths, align_right=True))
    print("-" * 128)

if __name__ == "__main__":
    print("正在统计代码行数与字符数...")
    cloc_json = run_cloc_json()
    data = process_cloc_data(cloc_json)
    if not data:
        print("错误：未解析到有效数据。")
        sys.exit(1)
    print_report(data)
