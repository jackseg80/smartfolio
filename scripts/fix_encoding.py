"""
Fix double-encoded UTF-8 (mojibake) in HTML files.

Problem: Files had UTF-8 content re-opened as CP1252 then re-saved as UTF-8,
causing double-encoding. Some bytes (0x81, 0x8D, 0x8F, 0x90, 0x9D) are
undefined in Python's cp1252 codec but valid in Windows-1252.

Strategy: Use extended CP1252 encoder that handles the 5 undefined bytes,
then decode as UTF-8 line-by-line.
"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

STATIC_DIR = Path(__file__).parent.parent / "static"

# 5 bytes undefined in Python's cp1252 but present in real Windows-1252 files.
# Python decodes them as C1 control chars U+0081..U+009D but refuses to encode back.
CP1252_UNDEFINED = {
    '\x81': b'\x81',
    '\x8d': b'\x8d',
    '\x8f': b'\x8f',
    '\x90': b'\x90',
    '\x9d': b'\x9d',
}

MOJIBAKE_MARKERS = [
    # French accents: Ã + second byte decoded as CP1252
    '\u00c3\u00a9', '\u00c3\u00a8', '\u00c3\u00a0', '\u00c3\u00a7',
    '\u00c3\u00af', '\u00c3\u00a2', '\u00c3\u00b4', '\u00c3\u00aa', '\u00c3\u00ae',
    # Smart quotes/dashes mojibake
    '\u00e2\u20ac\u2122', '\u00e2\u20ac\u0153',
    '\u00e2\u20ac\u201c', '\u00e2\u20ac\u201d',
    # Gear/info emoji fragments
    '\u00e2\u0161\u2122',  # âš™
    '\u00e2\u201e\u00b9',  # â„¹
    # Capital accents
    '\u00c3\u2030', '\u00c3\u20ac',
    # 4-byte emoji prefix: ð + Ÿ variants
    '\u00f0\u0178',
]


def smart_encode_cp1252(text):
    """Encode text to CP1252 bytes, handling 5 undefined positions."""
    result = bytearray()
    for char in text:
        if char in CP1252_UNDEFINED:
            result.extend(CP1252_UNDEFINED[char])
        else:
            result.extend(char.encode('cp1252'))
    return bytes(result)


def has_mojibake(text):
    """Check if text likely contains mojibake (double-encoded UTF-8 via CP1252)."""
    # Quick marker check for obvious patterns
    if any(m in text for m in MOJIBAKE_MARKERS):
        return True
    # Broader check: try roundtrip on first 50 non-ASCII lines
    count = 0
    for line in text.split('\n'):
        try:
            line.encode('ascii')
            continue
        except UnicodeEncodeError:
            pass
        count += 1
        if count > 50:
            break
        try:
            encoded = smart_encode_cp1252(line)
            fixed = encoded.decode('utf-8')
            if fixed != line:
                return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Try mixed approach
            fixed = fix_mixed_line(line)
            if fixed != line:
                return True
    return False


def fix_line(line):
    """Try to reverse double-encoding on a single line.

    Strategy: Try cp1252 roundtrip on every line. If it succeeds and
    produces different text, the line was mojibake. If it produces
    the same text or fails, keep the original.
    """
    # Fast path: pure ASCII lines can't be mojibake
    try:
        line.encode('ascii')
        return line, False
    except UnicodeEncodeError:
        pass

    # Try full-line roundtrip
    try:
        encoded = smart_encode_cp1252(line)
        fixed = encoded.decode('utf-8')
        if fixed != line:
            return fixed, True
        return line, False
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Line has chars outside cp1252 range - try mixed fix
        fixed = fix_mixed_line(line)
        if fixed != line:
            return fixed, True
        return line, False


def can_smart_encode_cp1252(char):
    """Check if a character can be encoded to CP1252 (including the 5 undefined)."""
    if char in CP1252_UNDEFINED:
        return True
    try:
        char.encode('cp1252')
        return True
    except UnicodeEncodeError:
        return False


def smart_encode_char(char):
    """Encode a single character to CP1252 byte(s)."""
    if char in CP1252_UNDEFINED:
        return CP1252_UNDEFINED[char]
    return char.encode('cp1252')


def fix_mixed_line(line):
    """Fix a line with mixed correct/double-encoded content."""
    result = []
    buffer = []

    for char in line:
        if can_smart_encode_cp1252(char):
            buffer.append(char)
        else:
            # Flush buffer
            if buffer:
                segment = ''.join(buffer)
                try:
                    encoded = smart_encode_cp1252(segment)
                    result.append(encoded.decode('utf-8'))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    result.append(segment)
                buffer = []
            result.append(char)

    if buffer:
        segment = ''.join(buffer)
        try:
            encoded = smart_encode_cp1252(segment)
            result.append(encoded.decode('utf-8'))
        except (UnicodeEncodeError, UnicodeDecodeError):
            result.append(segment)

    return ''.join(result)


def check_file(filepath):
    data = filepath.read_bytes()
    has_bom = data.startswith(b'\xef\xbb\xbf')
    text = data.decode('utf-8', errors='replace')
    mojibake = has_mojibake(text)
    return has_bom, mojibake, len(data)


def fix_file(filepath, dry_run=False):
    data = filepath.read_bytes()
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]

    text = data.decode('utf-8')
    lines = text.split('\n')
    fixed_lines = []
    changes = []

    for i, line in enumerate(lines):
        fixed, changed = fix_line(line)
        fixed_lines.append(fixed)
        if changed:
            changes.append((i + 1, line.strip()[:80], fixed.strip()[:80]))

    if dry_run:
        for lineno, old, new in changes[:10]:
            print(f"  L{lineno}: {old}")
            print(f"     -> {new}")
        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")
        print(f"  Total: {len(changes)} lines changed")
        return len(changes) > 0
    else:
        fixed_text = '\n'.join(fixed_lines)
        filepath.write_bytes(fixed_text.encode('utf-8'))
        return len(changes) > 0


def main():
    dry_run = '--dry-run' in sys.argv
    target_files = []
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            target_files.append(arg)

    if target_files:
        files = [Path(f) for f in target_files]
    else:
        files = sorted(STATIC_DIR.glob('*.html'))

    mode = 'DRY RUN' if dry_run else 'FIXING'
    print(f"{mode} - Checking {len(files)} files\n")

    fixed_count = 0
    for f in files:
        has_bom, mojibake, size = check_file(f)
        issues = []
        if has_bom:
            issues.append("BOM")
        if mojibake:
            issues.append("MOJIBAKE")

        if issues:
            print(f"\n{f.name}: {', '.join(issues)} ({size:,} bytes)")
            was_fixed = fix_file(f, dry_run=dry_run)
            if was_fixed and not dry_run:
                fixed_count += 1
                has_bom2, mojibake2, size2 = check_file(f)
                if not has_bom2 and not mojibake2:
                    print(f"  -> FIXED OK ({size2:,} bytes)")
                else:
                    print(f"  -> PARTIAL: BOM={has_bom2} MOJIBAKE={mojibake2} ({size2:,} bytes)")
            elif not was_fixed and not dry_run and has_bom:
                fixed_count += 1
                _, _, size2 = check_file(f)
                print(f"  -> BOM stripped ({size2:,} bytes)")
        else:
            print(f"{f.name}: OK")

    if dry_run:
        print(f"\nDry run complete. Use without --dry-run to apply.")
    else:
        print(f"\nDone. Fixed {fixed_count} files.")


if __name__ == '__main__':
    main()
