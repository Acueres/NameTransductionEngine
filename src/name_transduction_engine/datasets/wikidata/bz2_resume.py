"""bzip2 decoding that can start again in the middle of a ~100 GB stream.

A bzip2 stream is a header (`BZh9`) followed by blocks of up to 900 kB of
input each. Blocks are bit-aligned, not byte-aligned, and the standard
decoder can only start at a stream header. To resume, we:

1. find a block start (the 48-bit magic 0x314159265359) at any bit position
   in a few MiB of compressed bytes before the checkpoint;
2. hand the decoder a fresh header followed by the compressed bits from that
   block on, shifted left so the block starts on a byte boundary ("spliced"
   stream);
3. stop feeding that spliced stream just before its end. The decoder checks
   every block's own CRC as usual, but the whole-stream CRC at the end
   would not match (the spliced stream lacks the blocks before the
   checkpoint), so it is never read. The stop point is chosen so the last
   block is complete and the final CRC is not;
4. continue with the standard decoder at the next stream header (multi-stream
   files) or stop at the end of the file.

The caller locates its exact position again by content (see build.py), so
this module only has to start at or before the checkpoint.
"""

import bz2
import re

from typing import Final, Iterable, Iterator

BLOCK_MAGIC: Final[int] = 0x314159265359
_STREAM_HEADER_RE: Final = re.compile(rb"BZh[1-9](?:1AY&SY|\x17rE8P\x90)")
_HEADER_LEN: Final[int] = 10
_FRESH_HEADER: Final[bytes] = b"BZh9"


class Bz2DataError(RuntimeError):
    pass


def _magic_patterns() -> list[tuple[int, bytes, int, int, int, int]]:
    """For each bit shift s (0..7) of the magic inside a byte: the bytes that
    are fully determined, and masks to check the partial first/last bytes"""
    out = []
    for s in range(8):
        if s == 0:
            out.append((0, BLOCK_MAGIC.to_bytes(6, "big"), 0, 0, 0, 0))
            continue
        full = (BLOCK_MAGIC << (8 - s)).to_bytes(7, "big")
        first_mask = (1 << (8 - s)) - 1  # low bits of the first byte
        last_shift = 8 - s  # the last byte's top s bits
        out.append(
            (
                s,
                full[1:6],
                first_mask,
                full[0] & first_mask,
                last_shift,
                full[6] >> last_shift,
            )
        )
    return out


_PATTERNS: Final = _magic_patterns()


def find_block_starts(data: bytes, base_offset: int = 0) -> list[int]:
    """Absolute bit offsets of every block magic in `data`, sorted.

    `base_offset` is the file offset of data[0]. A random 48-bit match inside
    compressed data is possible but rare (about once per 30 TB); callers try
    the next candidate if decoding fails"""
    found: set[int] = set()
    for s, core, first_mask, first_val, last_shift, last_val in _PATTERNS:
        i = data.find(core)
        while i >= 0:
            if s == 0:
                found.add((base_offset + i) * 8)
            else:
                lo, hi = i - 1, i + 5
                if (
                    lo >= 0
                    and hi < len(data)
                    and data[lo] & first_mask == first_val
                    and data[hi] >> last_shift == last_val
                ):
                    found.add((base_offset + lo) * 8 + s)
            i = data.find(core, i + 1)
    return sorted(found)


class _Shifter:
    """Shift a byte stream left by s bits. For n input bytes it returns n-1
    output bytes; the last input byte is held back for the next call"""

    def __init__(self, shift: int) -> None:
        self.shift = shift
        self.carry = b""

    def feed(self, data: bytes) -> bytes:
        if self.shift == 0:
            return data
        buf = self.carry + data
        if len(buf) < 2:
            self.carry = buf
            return b""
        n = len(buf) - 1
        value = int.from_bytes(buf, "big") >> (8 - self.shift)
        self.carry = buf[-1:]
        return (value & ((1 << (8 * n)) - 1)).to_bytes(n, "big")


def decode(
    chunks: Iterable[tuple[int, bytes]], start_bit: int = 0, *, spliced: bool = False
) -> Iterator[tuple[bytes, int]]:
    """Decode compressed chunks into `(output, position)` pairs.

    `chunks` must start at byte `start_bit // 8`. `position` is the file
    offset of the first compressed byte not yet given to the decoder when the
    output was produced: every byte of `output` comes from blocks that lie
    entirely before it.

    `spliced=False`: `start_bit` is a byte-aligned stream header (usually 0).
    `spliced=True`: `start_bit` is the first bit of a block (see module doc).
    """
    iterator = iter(chunks)
    if spliced:
        leftover, position = yield from _decode_spliced(iterator, start_bit)
    else:
        if start_bit % 8:
            raise ValueError("a stream header is byte-aligned")
        leftover, position = b"", start_bit // 8
    yield from _decode_streams(iterator, leftover, position)


def _decode_spliced(
    chunks: Iterator[tuple[int, bytes]], start_bit: int
) -> Iterator[tuple[bytes, int]]:
    """Returns (bytes from the next stream header on, their file offset)"""
    shifter = _Shifter(start_bit % 8)
    decoder = bz2.BZ2Decompressor()
    pending = bytearray()
    pending_start = start_bit // 8
    search_from = 0
    started = False
    stream_end: int | None = None

    def feed(upto: int) -> Iterator[tuple[bytes, int]]:
        nonlocal pending_start, started
        n = upto - pending_start
        if n <= 0:
            return
        data = shifter.feed(bytes(pending[:n]))
        del pending[:n]
        pending_start += n
        if not started:
            data = _FRESH_HEADER + data
            started = True
        try:
            out = decoder.decompress(data)
        except OSError as exc:
            raise Bz2DataError(
                f"corrupt bzip2 data before byte {pending_start:,}: {exc}"
            ) from exc
        if decoder.eof:  # cannot happen: the final CRC is never fed
            raise Bz2DataError("spliced stream ended unexpectedly")
        if out:
            yield out, pending_start

    for offset, data in chunks:
        if offset != pending_start + len(pending):
            raise ValueError("chunks are not contiguous")
        pending += data
        match = _STREAM_HEADER_RE.search(pending, search_from)
        if match is not None:
            stream_end = pending_start + match.start()
            break
        # Keep a tail so a header split across chunks is still found
        yield from feed(pending_start + len(pending) - 16)
        search_from = max(0, len(pending) - _HEADER_LEN)

    if stream_end is None:  # end of file
        stream_end = pending_start + len(pending)

    # Stop one byte before the stream end: the last block is complete, the
    # end-of-stream CRC is not (see module doc)
    yield from feed(stream_end - 1)
    leftover = bytes(pending[stream_end - pending_start :])
    return leftover, stream_end + len(leftover)


def _decode_streams(
    chunks: Iterator[tuple[int, bytes]], leftover: bytes, position: int
) -> Iterator[tuple[bytes, int]]:
    decoder = bz2.BZ2Decompressor()
    fed_any = False

    def run(data: bytes, pos: int) -> Iterator[tuple[bytes, int]]:
        nonlocal decoder, fed_any
        while data:
            if not fed_any and not _could_be_header(data):
                raise Bz2DataError(
                    f"unexpected data after a bzip2 stream at byte {pos:,}"
                )
            fed_any = True
            try:
                out = decoder.decompress(data)
            except OSError as exc:
                raise Bz2DataError(
                    f"corrupt bzip2 data before byte {pos:,}: {exc}"
                ) from exc
            if out:
                yield out, pos
            if decoder.eof:
                data = decoder.unused_data
                decoder = bz2.BZ2Decompressor()
                fed_any = False
            else:
                data = b""

    if leftover:
        yield from run(leftover, position)
    for offset, data in chunks:
        if offset != position:
            raise ValueError("chunks are not contiguous")
        position += len(data)
        yield from run(data, position)

    if fed_any and not decoder.eof:
        raise Bz2DataError(f"bzip2 stream truncated at byte {position:,}")


def _could_be_header(data: bytes) -> bool:
    head = data[:3]
    return b"BZh".startswith(head) if len(head) < 3 else head == b"BZh"
