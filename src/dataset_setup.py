"""下载和整理工业异常数据，核心输入留在独立 Main 工程内。"""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.project_paths import RUNS_ROOT, relative_path, resolve_path


MVTEC_CATEGORIES = (
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
)
VISA_CATEGORIES = (
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
    'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
)
MVTEC_SHA256 = 'cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d'
SOURCES = {
    'mvtec_ad': {
        'url': 'https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz?download=true',
        'filename': 'mvtec_anomaly_detection.tar.xz',
        'size': 5264982680,
        'sha256': MVTEC_SHA256,
        'official_page': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
        'checksum_source': 'https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py',
        'license': 'CC-BY-NC-SA-4.0',
        'source_note': '官方旧链接返回404；镜像归档必须与上游公布的SHA256完全一致。',
    },
    'visa': {
        'url': 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar',
        'filename': 'VisA_20220922.tar',
        'size': 1929840640,
        'official_page': 'https://github.com/amazon-science/spot-diff',
        'license': 'CC-BY-4.0',
    },
}
VISA_SPLIT_URL = 'https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv'


def file_sha256(path):
    """流式校验大归档，避免把整个下载包读入内存。"""
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """持久化相对引用及有限数值，目录不存在时自动创建。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def download_ranges(url, partial, total):
    """八个连接领取小分段，避免个别慢连接长期占据大范围并保留已下载字节。"""
    folder = partial.with_name(partial.name + '.chunks')
    folder.mkdir(parents=True, exist_ok=True)
    size = min(64 * 1024 * 1024, (total + 7) // 8)
    count = (total + size - 1) // size
    legacy_size = (total + 7) // 8
    available = []
    legacy_folder = partial.with_name(partial.name + '.ranges')
    for index in range(8):
        old = legacy_folder / f'{index:02d}.part'
        if old.exists() and old.stat().st_size:
            available.append((index * legacy_size, index * legacy_size + old.stat().st_size, old))
    if partial.exists() and partial.stat().st_size < total:
        available.append((0, partial.stat().st_size, partial))
    for index in range(count):
        start, end = index * size, min((index + 1) * size, total)
        target = folder / f'{index:03d}.part'
        existing = target.stat().st_size if target.exists() else 0
        if existing == end - start:
            continue
        cursor, blocks = start, []
        while cursor < end:
            candidates = [item for item in available if item[0] <= cursor < item[1]]
            if not candidates:
                break
            source_start, source_end, source_path = max(candidates, key=lambda item: item[1])
            with source_path.open('rb') as source:
                source.seek(cursor - source_start)
                block = source.read(min(end, source_end) - cursor)
                if not block:
                    break
                blocks.append(block)
                cursor += len(block)
        if cursor - start > existing:
            target.write_bytes(b''.join(blocks))
    print(f'{partial.name}: resume {sum(path.stat().st_size for path in folder.glob("*.part")) / 1e9:.2f}/{total / 1e9:.2f} GB', flush=True)

    def fetch(index):
        start, end = index * size, min((index + 1) * size, total) - 1
        target = folder / f'{index:03d}.part'
        for attempt in range(6):
            existing = target.stat().st_size if target.exists() else 0
            if existing == end - start + 1:
                return target
            if existing > end - start + 1:
                raise ValueError('续传分段超过预期范围')
            current = start + existing
            request = Request(url, headers={'Range': f'bytes={current}-{end}',
                                           'User-Agent': 'industrial-anomaly-course/1.0'})
            try:
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {current}-'):
                        raise ValueError('来源不支持精确的分段续传')
                    with target.open('ab') as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                if target.stat().st_size != end - start + 1:
                    raise OSError('下载分段尚未完整')
                print(f'{partial.name}: range {index + 1}/{count} complete', flush=True)
                return target
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(min(attempt + 1, 5))
        raise RuntimeError('分段下载未完成')

    with ThreadPoolExecutor(max_workers=min(8, count)) as pool:
        parts = list(pool.map(fetch, range(count)))
    with partial.open('wb') as output:
        for path in parts:
            with path.open('rb') as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)


def download_file(url, destination, *, expected_size=None, expected_sha256=None):
    """按字节范围恢复下载，完整性通过后才将临时文件发布为正式归档。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = file_sha256(destination)
        if ((expected_size is not None and destination.stat().st_size != expected_size)
                or (expected_sha256 is not None and digest != expected_sha256)):
            raise ValueError(f'已有下载校验失败，保留文件供检查: {destination.name}')
        return dict(bytes=destination.stat().st_size, sha256=digest)
    partial = destination.with_name(destination.name + '.part')
    if expected_size is not None and expected_size > 64 * 1024 * 1024:
        if not partial.exists() or partial.stat().st_size != expected_size:
            download_ranges(url, partial, expected_size)
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset == expected_size:
            break
        request = Request(url, headers={'User-Agent': 'industrial-anomaly-course/1.0',
                                      **({'Range': f'bytes={offset}-'} if offset else {})})
        try:
            with urlopen(request, timeout=45) as response:
                if offset and response.status != 206:
                    offset = 0
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('服务器返回的续传范围与本地临时文件不一致')
                total = expected_size or (offset + int(response.headers.get('Content-Length', 0)))
                reported, started, initial = time.monotonic(), time.monotonic(), offset
                with partial.open('ab' if offset else 'wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - reported >= 25:
                            speed = (offset - initial) / max(time.monotonic() - started, 1)
                            print(f'{destination.name}: {offset / 1e9:.2f}/{total / 1e9:.2f} GB, {speed / 1e6:.1f} MB/s', flush=True)
                            reported = time.monotonic()
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError('下载连接结束但文件长度未达到来源公布值')
            break
        except (OSError, HTTPError) as error:
            print(f'{destination.name}: retry {attempt + 1}, {type(error).__name__}', flush=True)
            if attempt == 5:
                raise
            time.sleep(min(attempt + 1, 5))
    digest = file_sha256(partial)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'归档SHA256不匹配，保留临时文件: {destination.name}')
    partial.replace(destination)
    print(f'{destination.name}: download verified, {destination.stat().st_size} bytes', flush=True)
    return dict(bytes=destination.stat().st_size, sha256=digest)


def download_dataset(name, cache_root):
    """下载公开数据及官方划分，并记录实际来源与完整性校验值。"""
    source = SOURCES[name]
    folder = Path(cache_root) / name
    archive = folder / source['filename']
    print(f'{name}: starting/resuming download', flush=True)
    integrity = download_file(source['url'], archive, expected_size=source['size'],
                              expected_sha256=source.get('sha256'))
    result = dict(source, **integrity, archive=relative_path(archive))
    if name == 'visa':
        split = folder / '1cls.csv'
        result['split_integrity'] = download_file(VISA_SPLIT_URL, split)
        result['split_url'] = VISA_SPLIT_URL
    write_json(folder / 'download.json', result)
    return result


def download_datasets(names, cache_root=RUNS_ROOT / 'downloads'):
    """独立下载两套归档，减少等待而不并行修改核心数据目录。"""
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        return dict(zip(names, pool.map(lambda name: download_dataset(name, cache_root), names)))


def safe_member_path(name):
    """归档和划分中的路径只能指向其数据根目录内部。"""
    value = PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError(f'非法归档路径: {name}')
    return Path(*value.parts)


def extract_archive(archive, destination):
    """只展开普通文件和目录，不接受链接或跨目录成员。"""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r|*') as source:
        for member in source:
            path = destination / safe_member_path(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, path.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            else:
                raise ValueError(f'不支持的归档成员: {member.name}')


def merge_preserving_existing(source, destination):
    """原四类只核对不覆盖，存在内容冲突时明确失败。"""
    source, destination = Path(source), Path(destination)
    files = sorted(path for path in source.rglob('*') if path.is_file())
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists() and (target.stat().st_size != path.stat().st_size
                                or file_sha256(target) != file_sha256(path)):
            raise ValueError(f'已有核心数据与归档内容冲突: {relative_path(target)}')
    added, retained = 0, 0
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists():
            retained += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        added += 1
    return dict(added_files=added, retained_identical_files=retained)


def read_visa_split(split_file):
    """验证官方单类划分，禁止异常训练、重复样本和目标文件名冲突。"""
    with Path(split_file).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['object', 'split', 'label', 'image', 'mask']:
            raise ValueError('VisA划分字段与官方1cls.csv不一致')
        rows = list(reader)
    sources, targets = set(), set()
    for row in rows:
        category, split, label = row['object'], row['split'], row['label']
        if category not in VISA_CATEGORIES or split not in ('train', 'test') or label not in ('normal', 'anomaly'):
            raise ValueError('VisA包含未知类别、划分或标签')
        if split == 'train' and label != 'normal':
            raise ValueError('VisA官方单类训练集不得含异常样本')
        image = safe_member_path(row['image'])
        if not image.parts or image.parts[0] != category:
            raise ValueError('VisA源图与类别不一致')
        if label == 'anomaly':
            mask = safe_member_path(row['mask'])
            if not row['mask'] or mask.parts[0] != category:
                raise ValueError('VisA异常样本缺少同类掩膜')
        elif row['mask']:
            raise ValueError('VisA正常样本不应带有缺陷掩膜')
        kind = 'good' if label == 'normal' else 'bad'
        target = (category, split, kind, image.stem)
        if row['image'] in sources or target in targets:
            raise ValueError('VisA官方划分存在重复源图或目标路径冲突')
        sources.add(row['image'])
        targets.add(target)
    return rows


def convert_visa(raw_root, split_file, destination):
    """保留官方训练测试划分和原尺寸，将图像与掩膜转为加载器兼容格式。"""
    import numpy as np
    from PIL import Image

    raw_root, destination = Path(raw_root), Path(destination)
    rows = read_visa_split(split_file)

    def convert(row):
        category, split = row['object'], row['split']
        kind = 'good' if row['label'] == 'normal' else 'bad'
        source = raw_root / safe_member_path(row['image'])
        image_path = destination / category / split / kind / (source.stem + '.png')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as original:
            image = original.convert('RGB')
            size = image.size
            image.save(image_path, compress_level=3)
        mask_path = None
        if kind == 'bad':
            with Image.open(raw_root / safe_member_path(row['mask'])) as original:
                if original.size != size:
                    raise ValueError(f'VisA掩膜尺寸不一致: {row["image"]}')
                mask = np.asarray(original)
                if mask.ndim != 2:
                    raise ValueError('VisA掩膜必须是单通道类别索引')
                binary = np.where(mask > 0, 255, 0).astype(np.uint8)
            mask_path = destination / category / 'ground_truth' / 'bad' / (source.stem + '_mask.png')
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary).save(mask_path)
        return dict(official_image=row['image'], official_mask=row['mask'] or None,
                    image_path=image_path.relative_to(destination).as_posix(),
                    mask_path=mask_path.relative_to(destination).as_posix() if mask_path else None,
                    category=category, split=split, label=0 if kind == 'good' else 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        mapping = list(pool.map(convert, rows))
    return mapping


def preserve_visa_annotations(raw_root, destination):
    """保留官方原始多类掩膜与类别CSV，二值化只作用于当前模型输入。"""
    raw_root, destination = Path(raw_root), Path(destination)
    retained = destination / '_meta' / 'source_annotations'
    rows = read_visa_split(destination / '_meta/1cls.csv')
    paths = [raw_root / category / 'image_anno.csv' for category in VISA_CATEGORIES]
    paths.extend(raw_root / safe_member_path(row['mask']) for row in rows if row['mask'])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'缺少官方VisA标注: {path.relative_to(raw_root).as_posix()}')
        target = retained / path.relative_to(raw_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def audit_dataset(name, *, root=None, decode=True, enforce_official_counts=True):
    """全量检查图片、掩膜及正常留出隔离，只作数据核验而不训练或调参。"""
    import cv2
    import numpy as np
    from src.data import load_dataset
    from src.image_io import read_image

    root = Path(root) if root is not None else resolve_path(f'data/{name}')
    categories = MVTEC_CATEGORIES if name == 'mvtec_ad' else VISA_CATEGORIES
    if not enforce_official_counts:
        categories = tuple(path.name for path in sorted(root.iterdir()) if (path / 'train').is_dir())
    config = dict(dataset=dict(root=relative_path(root), categories=list(categories),
                               validation_ratio=.2, random_seed=42))
    datasets = load_dataset(config)

    def inspect_category(item):
        category, splits = item
        unexpected = [path.name for path in (root / category / 'train').iterdir()
                      if path.is_dir() and path.name != 'good']
        if unexpected:
            raise ValueError(f'{category}训练目录包含非正常类别: {unexpected}')
        paths = [row['image_path'] for split in splits.values() for row in split]
        if len(paths) != len(set(paths)):
            raise ValueError(f'{category}训练、验证、测试发生路径重叠')
        if any(row['label'] != 0 or row['mask_path'] is not None
               for split in ('train', 'val') for row in splits[split]):
            raise ValueError('训练和校准只允许正常无掩膜样本')
        for split in splits.values():
            for row in split:
                if not decode:
                    continue
                image = read_image(row['image_path'], cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f'图像解码失败: {row["image_path"]}')
                if row['mask_path']:
                    mask = read_image(row['mask_path'], cv2.IMREAD_UNCHANGED)
                    if mask is None or mask.ndim != 2 or mask.shape != image.shape[:2]:
                        raise ValueError(f'掩膜尺寸或解码错误: {row["mask_path"]}')
                    if not np.isin(mask, (0, 255)).all() or not np.any(mask):
                        raise ValueError(f'缺陷掩膜应为非空二值标注: {row["mask_path"]}')
        return category, dict(train=len(splits['train']), val=len(splits['val']),
                              official_train=len(splits['train']) + len(splits['val']),
                              test_normal=sum(row['label'] == 0 for row in splits['test']),
                              test_anomaly=sum(row['label'] == 1 for row in splits['test']))

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = dict(pool.map(inspect_category, datasets.items()))
    total = dict(Counter({key: sum(row[key] for row in counts.values()) for key in next(iter(counts.values()))}))
    total['images'] = total['official_train'] + total['test_normal'] + total['test_anomaly']
    expected = (3629, 467, 1258) if name == 'mvtec_ad' else (8659, 962, 1200)
    actual = (total['official_train'], total['test_normal'], total['test_anomaly'])
    if enforce_official_counts and actual != expected:
        raise ValueError(f'{name}官方划分数量不一致: actual={actual}, expected={expected}')
    return dict(dataset=name, root=relative_path(root), categories=counts, totals=total,
                validation_ratio=.2, random_seed=42, decoded_all_images=decode,
                normal_only_training=True, threshold_source='normal_validation',
                errors=[], skipped_images=[], splits=datasets)


def prepare_dataset(name, cache_root, provenance):
    """先在辅助目录完成整理校验，再合并发布到核心数据目录。"""
    cache = Path(cache_root) / name
    stage = RUNS_ROOT / 'tmp' / 'dataset_expansion' / name
    stage.mkdir(parents=True, exist_ok=True)
    raw = stage / 'raw'
    marker = stage / 'extracted.json'
    if not marker.exists():
        print(f'{name}: extracting archive', flush=True)
        extract_archive(cache / SOURCES[name]['filename'], raw)
        write_json(marker, dict(archive_sha256=provenance['sha256']))
    elif json.loads(marker.read_text(encoding='utf-8'))['archive_sha256'] != provenance['sha256']:
        raise ValueError('展开缓存对应的归档已变化，请指定干净的准备目录')
    source = raw
    mapping = None
    if name == 'visa':
        print('visa: converting official one-class split to PNG', flush=True)
        source = stage / 'organized'
        mapping = convert_visa(raw, cache / '1cls.csv', source)
    print(f'{name}: validating staged dataset', flush=True)
    audit_dataset(name, root=source)
    destination = resolve_path(f'data/{name}')
    merge = merge_preserving_existing(source, destination)
    meta = destination / '_meta'
    meta.mkdir(parents=True, exist_ok=True)
    if name == 'visa':
        shutil.copy2(cache / '1cls.csv', meta / '1cls.csv')
        write_json(meta / 'official_mapping.json', mapping)
        preserve_visa_annotations(raw, destination)
    write_json(meta / 'provenance.json', dict(provenance, preparation=merge,
               format='MVTec-compatible PNG; masks binary 0/255',
               validation_protocol='20% normal official training holdout; seed=42',
               image_conversion='RGB decoded at original size, no resampling' if name == 'visa' else 'original files preserved'))
    report = audit_dataset(name, decode=False)
    report['decoded_all_images'] = True
    report['decode_evidence'] = 'Staged dataset decoded before byte-identical publication'
    write_json(meta / 'inventory.json', report)
    print(f'{name}: ready, {report["totals"]}', flush=True)
    return report
