#!/usr/bin/env python3
"""Converts a BB_RT_DUMP_TRIGGER dump (docs/path_tracing.md, step 2) into PNG files.

    python tools/rt/rt_dump.py <dump dir> [--out <dir>] [--index N]

Each raw image <dump>/dNNN_<name>_<w>x<h>_<format>.raw becomes <out>/dNNN_<name>.png (RGB,
HDR formats tone mapped) and one grayscale PNG per channel (<name>.r.png ...), and a line of
per-channel statistics. dNNN_sheet.png puts all of them on one labelled page. The scene
constants (dNNN_constants_864.bin) are printed as a table. Needs numpy and Pillow.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAW = re.compile(r'^d(\d{3})_(.+)_(\d+)x(\d+)_([A-Za-z0-9]+)\.raw$')

# Known scene constant ranges (docs/upscaler.md, "Scene constants").
CONSTANT_NOTES = {
    0: 'far plane, 1/far', 4: 'render width, height, 1/w, 1/h',
    8: 'view matrix rows (3x4)', 36: 'inverse projection', 52: 'projection (rows 52-67)',
    112: 'shadow cascade matrices (112-175)', 176: '2, 8, 15, 0', 180: 'inverse view (3x4)',
}


def unpack_float(bits, exponent_bits, mantissa_bits):
    """Unsigned small floats (B10G11R11): 5-bit exponent, bias 15."""
    mantissa = (bits & ((1 << mantissa_bits) - 1)).astype(np.float32)
    exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    scale = float(1 << mantissa_bits)
    normal = np.ldexp(1.0 + mantissa / scale, exponent.astype(np.int32) - 15)
    denormal = np.ldexp(mantissa / scale, -14)
    value = np.where(exponent == 0, denormal, normal)
    return np.where(exponent == 31, np.inf, value).astype(np.float32)


def decode(data, width, height, fmt):
    """Returns (pixels HxWxC float32 or uint8, channel names, kind)."""
    n = width * height
    if fmt in ('R8G8B8A8Unorm', 'R8G8B8A8Srgb', 'B8G8R8A8Unorm', 'B8G8R8A8Srgb', 'R8G8B8A8Snorm',
               'R8G8B8A8Uint'):
        px = np.frombuffer(data, np.uint8, n * 4).reshape(height, width, 4)
        if fmt.startswith('B8G8R8A8'):
            px = px[..., [2, 1, 0, 3]]
        return px, 'rgba', 'ldr'
    if fmt == 'B10G11R11UfloatPack32':
        bits = np.frombuffer(data, np.uint32, n).reshape(height, width)
        r = unpack_float(bits & 0x7FF, 5, 6)
        g = unpack_float((bits >> 11) & 0x7FF, 5, 6)
        b = unpack_float((bits >> 22) & 0x3FF, 5, 5)
        return np.stack([r, g, b], -1), 'rgb', 'hdr'
    if fmt == 'R16G16B16A16Sfloat':
        px = np.frombuffer(data, np.float16, n * 4).reshape(height, width, 4).astype(np.float32)
        return px, 'rgba', 'hdr'
    if fmt == 'R32G32Sfloat':
        px = np.frombuffer(data, np.float32, n * 2).reshape(height, width, 2)
        return px, 'rg', 'hdr'
    if fmt == 'R16G16Sfloat':
        px = np.frombuffer(data, np.float16, n * 2).reshape(height, width, 2).astype(np.float32)
        return px, 'rg', 'hdr'
    if fmt in ('R32Sfloat', 'D32Sfloat'):
        px = np.frombuffer(data, np.float32, n).reshape(height, width, 1)
        return px, 'd' if fmt == 'D32Sfloat' else 'r', 'depth' if fmt == 'D32Sfloat' else 'hdr'
    if fmt == 'S8Uint':
        return np.frombuffer(data, np.uint8, n).reshape(height, width, 1), 's', 'stencil'
    if fmt == 'A2B10G10R10UnormPack32':
        bits = np.frombuffer(data, np.uint32, n).reshape(height, width)
        chans = [(bits >> s) & 0x3FF for s in (0, 10, 20)] + [bits >> 30]
        px = np.stack([c.astype(np.float32) / m for c, m in zip(chans, (1023, 1023, 1023, 3))], -1)
        return px, 'rgba', 'unorm'
    raise ValueError(f'unknown format {fmt}')


def to_srgb8(linear):
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


def tonemap(rgb):
    finite = np.where(np.isfinite(rgb), rgb, 0.0)
    lum = finite.mean(-1)
    reference = np.percentile(lum[lum > 0], 99) if np.any(lum > 0) else 1.0
    x = finite * (0.8 / max(reference, 1e-6))
    return to_srgb8(x / (1.0 + x) * 1.6)


def gray(channel, kind):
    if kind in ('ldr',):
        return channel.astype(np.uint8)
    finite = np.where(np.isfinite(channel), channel, 0.0)
    lo, hi = np.percentile(finite, 0.5), np.percentile(finite, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((finite - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def stats(px, names):
    parts = []
    for i, name in enumerate(names):
        c = px[..., i].astype(np.float64)
        finite = c[np.isfinite(c)]
        if finite.size == 0:
            parts.append(f'{name}: no finite values')
            continue
        parts.append(f'{name} {finite.min():.4g}..{finite.max():.4g} mean {finite.mean():.4g}'
                     f' uniq {min(np.unique(finite[:200000]).size, 9999)}')
    return '; '.join(parts)


def linear_depth(depth, constants):
    """View-space distance from D3D depth (depth = zs + zo / z), see CameraMotion::Near."""
    if constants is None:
        return None
    zs, zo = constants[62], constants[63]
    with np.errstate(divide='ignore', invalid='ignore'):
        z = zo / (depth - zs)
    return np.where((depth > 0) & (depth < 1), z, np.nan)


def validate_rays(px, stem, sheet_items):
    """rt_validate (step 4): r = view z of the ray's hit in the acceleration structure (0: none),
    g = view z of the depth buffer (0: sky). Green: they agree within 2%; red: the ray hit
    something nearer (extra geometry, e.g. alpha-tested leaves traced as full quads); blue: the
    ray went past the visible surface or missed it (geometry missing from the structure)."""
    hit, depth = px[..., 0], px[..., 1]
    surface = depth > 0
    match = surface & (hit > 0) & (np.abs(hit - depth) <= 0.02 * depth + 0.01)
    nearer = surface & (hit > 0) & (hit < depth * 0.98 - 0.01)
    missing = surface & ~match & ~nearer
    sky_hit = ~surface & (hit > 0)
    total = max(int(surface.sum()), 1)
    print(f'rt_validate: of {total} surface pixels {match.sum() / total:.1%} match the depth, '
          f'{nearer.sum() / total:.1%} hit nearer, {missing.sum() / total:.1%} missing; '
          f'sky pixels with a hit: {sky_hit.mean():.1%}')
    if match.any():
        error = np.abs(hit - depth)[match] / depth[match]
        print(f'{"":12s} matching pixels: median relative error {np.median(error):.2e}')
    image = np.zeros(px.shape[:2] + (3,), np.uint8)
    image[match] = (40, 200, 60)
    image[nearer] = (220, 50, 40)
    image[missing] = (50, 90, 230)
    image[sky_hit] = (200, 200, 60)
    Image.fromarray(image).save(f'{stem}.png')
    sheet_items.append(('rt_validate: green match, red nearer, blue missing', image))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dump', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--index', type=int, help='dump number (default: the last one)')
    args = parser.parse_args()
    out = args.out or args.dump / 'png'
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.dump.iterdir() if RAW.match(p.name))
    if not files:
        sys.exit(f'no dNNN_*.raw files in {args.dump}')
    index = args.index if args.index is not None else max(int(RAW.match(p.name)[1]) for p in files)
    files = [p for p in files if int(RAW.match(p.name)[1]) == index]
    constants_file = args.dump / f'd{index:03d}_constants_864.bin'
    constants = np.fromfile(constants_file, np.float32) if constants_file.is_file() else None

    sheet_items = []
    scene_steps = []  # scene color snapshots: before each full-screen pass, then the end
    for path in files:
        _, name, w, h, fmt = RAW.match(path.name).groups()
        w, h = int(w), int(h)
        px, names, kind = decode(path.read_bytes(), w, h, fmt)
        print(f'{name:9s} {w}x{h} {fmt:24s} {stats(px, names)}')
        stem = out / f'd{index:03d}_{name}'
        if kind == 'depth':
            # The shadow map has the light's projection, not the camera's.
            z = linear_depth(px[..., 0], constants) if name == 'depth' else None
            if z is not None:
                finite = z[np.isfinite(z)]
                if finite.size:
                    print(f'{"":9s} view distance {finite.min():.3g}..{finite.max():.3g}'
                          f' median {np.median(finite):.3g} (sky/cleared: {np.isnan(z).mean():.1%})')
                shown = np.where(np.isfinite(z), np.log(np.clip(z, 0.05, 3000)), np.log(3000))
                image = gray(shown[..., None], 'hdr')[..., 0]
            else:
                image = gray(px, 'hdr')[..., 0]
            Image.fromarray(image).save(f'{stem}.png')
            sheet_items.append((f'{name} (log distance)' if z is not None else name, image))
            continue
        if kind == 'stencil':
            values, counts = np.unique(px, return_counts=True)
            print(f'{"":9s} values: ' + ', '.join(f'{v:#04x} {c / px.size:.1%}'
                                                   for v, c in zip(values, counts) if c / px.size > 0.001))
            palette = np.random.default_rng(1).integers(40, 255, (256, 3), dtype=np.uint8)
            palette[0] = 0
            image = palette[px[..., 0]]
            Image.fromarray(image).save(f'{stem}.png')
            sheet_items.append((f'{name} (values coloured)', image))
            continue
        if name == 'rt_validate':
            validate_rays(px, stem, sheet_items)
            continue
        rgb = px[..., :3] if px.shape[-1] >= 3 else np.repeat(px[..., :1], 3, -1)
        if name.startswith('scene_'):
            scene_steps.append((name, rgb))
        image = tonemap(rgb) if kind == 'hdr' else (rgb if kind == 'ldr' else to_srgb8(rgb))
        Image.fromarray(image).save(f'{stem}.png')
        sheet_items.append((f'{name} {fmt} rgb', image))
        for i, channel in enumerate(names):
            plane = gray(px[..., i], kind)
            Image.fromarray(plane).save(f'{stem}.{channel}.png')
            sheet_items.append((f'{name}.{channel}', plane))

    # What each full-screen pass over the scene did (the fog composite among them): the next
    # snapshot minus this one, and the per-pixel ratio (below 1: the pass dimmed the scene, as
    # fog transmittance does; above: it added light).
    order = {name: i for i, name in enumerate(sorted(n for n, _ in scene_steps if n != 'scene_end'))}
    order['scene_end'] = len(order)
    scene_steps.sort(key=lambda step: order[step[0]])
    for (before_name, before), (after_name, after) in zip(scene_steps, scene_steps[1:]):
        added = np.where(np.isfinite(after - before), after - before, 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = after.mean(-1) / before.mean(-1)
        ratio = ratio[np.isfinite(ratio)]
        changed = np.abs(added).max(-1) > 1e-4
        print(f'{before_name} -> {after_name}: {changed.mean():.1%} of pixels changed, added'
              f' mean {added.mean():.4g} max {added.max():.4g}, ratio median'
              f' {np.median(ratio) if ratio.size else float("nan"):.3g}')
        image = tonemap(np.abs(added))
        label = f'{before_name}_to_{after_name}'
        Image.fromarray(image).save(out / f'd{index:03d}_{label}.png')
        sheet_items.append((f'{label} |difference|', image))

    if constants is not None:
        print('\nscene constants (floats):')
        for i in range(0, len(constants), 4):
            note = next((v for k, v in CONSTANT_NOTES.items() if k == i), '')
            row = ' '.join(f'{v:12.6g}' for v in constants[i:i + 4])
            print(f'  [{i:3d}] {row}  {note}')

    # Contact sheet: thumbnails in rows of four.
    thumb_w, thumb_h, label_h, columns = 480, 270, 18, 4
    rows = (len(sheet_items) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(sheet_items):
        x, y = (i % columns) * thumb_w, (i // columns) * (thumb_h + label_h)
        thumb = Image.fromarray(image).convert('RGB').resize((thumb_w, thumb_h), Image.BILINEAR)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 4, y + 3), label, fill=(230, 230, 230))
    sheet_path = out / f'd{index:03d}_sheet.png'
    sheet.save(sheet_path)
    print(f'\n{len(sheet_items)} images, sheet: {sheet_path}')


if __name__ == '__main__':
    main()
