"""HWP 5.0 문서에서 본문 텍스트를 추출한다.

한컴 COM 자동화(HWPFrame.HwpObject)는 보안 모듈 미등록으로 Open()에서 무한 대기하므로
사용하지 않는다. 대신 OLE 복합문서를 직접 파싱한다. (CLAUDE.md §4 참조)

    CFB → BodyText/SectionN → raw deflate 해제 → HWPTAG_PARA_TEXT(67) 추출
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import olefile

# 레코드 헤더 uint32: tag=h&0x3FF, level=(h>>10)&0x3FF, size=(h>>20)&0xFFF
HWPTAG_BEGIN = 0x010
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51  # 67

# 본문 중 제어문자 분류. 확장/인라인 제어문자는 16바이트(8 wchar)를 차지한다.
_WIDE_CTRL = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
_BREAK_CTRL = frozenset([10, 13])


def _is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    (flags,) = struct.unpack_from("<I", header, 36)
    return bool(flags & 0x01)


def _read_section(ole: olefile.OleFileIO, path: list[str], compressed: bool) -> bytes:
    raw = ole.openstream(path).read()
    if not compressed:
        return raw
    # HWP는 zlib 헤더 없는 raw deflate를 쓴다.
    return zlib.decompress(raw, -15)


def _parse_para_text(payload: bytes) -> str:
    out: list[str] = []
    i = 0
    n = len(payload)
    while i + 1 < n:
        (code,) = struct.unpack_from("<H", payload, i)
        if code in _BREAK_CTRL:
            out.append("\n")
            i += 2
        elif code == 9:
            out.append("\t")
            i += 16
        elif code < 32:
            i += 16 if code in _WIDE_CTRL else 2
        else:
            out.append(chr(code))
            i += 2
    return "".join(out)


def _walk_records(data: bytes) -> str:
    out: list[str] = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        (header,) = struct.unpack_from("<I", data, pos)
        pos += 4
        tag = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            (size,) = struct.unpack_from("<I", data, pos)
            pos += 4
        if pos + size > n:
            break
        if tag == HWPTAG_PARA_TEXT:
            out.append(_parse_para_text(data[pos : pos + size]))
            out.append("\n")
        pos += size
    return "".join(out)


def extract(hwp_path: Path) -> str:
    with olefile.OleFileIO(str(hwp_path)) as ole:
        compressed = _is_compressed(ole)
        sections = sorted(
            (e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"),
            key=lambda e: int(e[1].removeprefix("Section")),
        )
        if not sections:
            raise RuntimeError(f"BodyText 섹션을 찾지 못했습니다: {hwp_path}")
        return "\n".join(_walk_records(_read_section(ole, s, compressed)) for s in sections)


def to_markdown(hwp_path: Path) -> str:
    lines = [ln.strip() for ln in extract(hwp_path).splitlines()]
    body = "\n".join(ln for ln in lines if ln)
    return f"# {hwp_path.stem}\n\n> 원본: `{hwp_path.name}`\n> 추출: `tools/extract_hwp.py`\n\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="HWP 본문 텍스트 추출")
    ap.add_argument("hwp", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="출력 마크다운 경로 (생략 시 stdout)")
    args = ap.parse_args()

    md = to_markdown(args.hwp)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(md)} chars)")
    else:
        print(md)


if __name__ == "__main__":
    main()
