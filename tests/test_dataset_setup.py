"""验证官方划分、正常留出、数据转换与下载发布的关键边界。"""

from copy import deepcopy
import csv
from hashlib import sha256
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image
import pytest
import yaml

from src import dataset_setup
from src.data import load_dataset
from src.dataset_setup import (audit_dataset, convert_visa, download_file, download_ranges,
                               extract_archive, merge_preserving_existing, read_visa_split)
from src.project_paths import PROJECT_ROOT, resolve_path


@pytest.fixture
def visa_fixture(tmp_path):
    """按官方字段构造正常训练、正常测试和带多类索引的异常掩膜。"""
    raw = tmp_path / 'raw'
    rows = []
    rng = np.random.default_rng(7)
    for index, (split, label) in enumerate([('train', 'normal')] * 4 + [('test', 'normal'), ('test', 'anomaly')]):
        kind = 'Normal' if label == 'normal' else 'Anomaly'
        source = f'candle/Data/Images/{kind}/{index:04d}.JPG'
        image_path = raw / source
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, size=(13, 19, 3), dtype=np.uint8)).save(image_path)
        mask = ''
        if label == 'anomaly':
            mask = f'candle/Data/Masks/Anomaly/{index:04d}.png'
            mask_path = raw / mask
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            values = np.zeros((13, 19), np.uint8)
            values[2:4, 3:5] = 1
            values[6:8, 8:10] = 3
            Image.fromarray(values).save(mask_path)
        rows.append(dict(object='candle', split=split, label=label, image=source, mask=mask))
    split_file = tmp_path / '1cls.csv'
    write_split(split_file, rows)
    return raw, split_file, rows


def write_split(path, rows):
    """为异常划分约束测试写入与官方一致的字段顺序。"""
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['object', 'split', 'label', 'image', 'mask'])
        writer.writeheader()
        writer.writerows(rows)


def test_conversion_preserves_decoded_pixels_masks_and_official_split(visa_fixture, tmp_path):
    """不缩放或重新划分，图像像素与官方JPEG解码结果完全一致。"""
    raw, split_file, rows = visa_fixture
    destination = tmp_path / 'visa'
    mapping = convert_visa(raw, split_file, destination)
    assert len(mapping) == len(rows)
    for original, result in zip(rows, mapping):
        with Image.open(raw / original['image']) as source, Image.open(destination / result['image_path']) as target:
            np.testing.assert_array_equal(np.asarray(source.convert('RGB')), np.asarray(target))
        assert result['split'] == original['split']
        assert not Path(result['image_path']).is_absolute()
    defect = next(row for row in mapping if row['label'] == 1)
    binary = np.asarray(Image.open(destination / defect['mask_path']))
    assert set(np.unique(binary)) == {0, 255}
    original_mask = np.asarray(Image.open(raw / rows[-1]['mask']))
    np.testing.assert_array_equal(binary > 0, original_mask > 0)
    assert Path(defect['mask_path']).name == '0005_mask.png'
    config = dict(dataset=dict(root=str(destination), categories=['candle'], validation_ratio=.25, random_seed=42))
    splits = load_dataset(config)['candle']
    assert [len(splits[key]) for key in ('train', 'val', 'test')] == [3, 1, 2]
    assert all(row['label'] == 0 and row['mask_path'] is None for key in ('train', 'val') for row in splits[key])
    report = audit_dataset('visa', root=destination, enforce_official_counts=False)
    assert report['totals']['images'] == 6
    assert not report['errors']


@pytest.mark.parametrize('bad_case', ['anomaly_train', 'duplicate', 'traversal', 'missing_mask'])
def test_invalid_official_split_is_rejected(visa_fixture, bad_case):
    """错误划分必须在转换前失败，不能把缺陷或重复图混入正常训练。"""
    _, split_file, original = visa_fixture
    rows = deepcopy(original)
    if bad_case == 'anomaly_train':
        rows[0]['label'] = 'anomaly'
    elif bad_case == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif bad_case == 'traversal':
        rows[0]['image'] = '../outside.JPG'
    else:
        rows[-1]['mask'] = ''
    write_split(split_file, rows)
    with pytest.raises(ValueError):
        read_visa_split(split_file)


def test_archive_traversal_is_rejected(tmp_path):
    """下载归档不能写到准备目录外部。"""
    archive = tmp_path / 'bad.tar'
    with tarfile.open(archive, 'w') as target:
        member = tarfile.TarInfo('../escape.txt')
        member.size = 3
        target.addfile(member, io.BytesIO(b'bad'))
    with pytest.raises(ValueError, match='非法'):
        extract_archive(archive, tmp_path / 'stage')
    assert not (tmp_path / 'escape.txt').exists()


def test_existing_core_data_conflict_prevents_any_merge(tmp_path):
    """先预检全部冲突，避免合并一半后破坏原四类核心数据。"""
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    destination.mkdir()
    (source / 'new.png').write_bytes(b'new')
    (source / 'old.png').write_bytes(b'changed')
    (destination / 'old.png').write_bytes(b'original')
    with pytest.raises(ValueError, match='冲突'):
        merge_preserving_existing(source, destination)
    assert (destination / 'old.png').read_bytes() == b'original'
    assert not (destination / 'new.png').exists()


class Response(io.BytesIO):
    """模拟明确返回字节范围的下载连接。"""

    def __init__(self, payload, start, end, total):
        super().__init__(payload)
        self.status = 206
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}', 'Content-Length': str(len(payload))}


def test_resumable_download_checks_range_and_checksum(monkeypatch, tmp_path):
    """已有前缀只能追加匹配范围，校验通过才发布正式文件。"""
    destination = tmp_path / 'archive.bin'
    destination.with_name('archive.bin.part').write_bytes(b'abc')

    def open_response(request, timeout):
        """服务端只返回缺失的后半段。"""
        assert request.get_header('Range') == 'bytes=3-'
        return Response(b'def', 3, 5, 6)

    monkeypatch.setattr(dataset_setup, 'urlopen', open_response)
    result = download_file('https://example.test/data', destination, expected_size=6,
                           expected_sha256=sha256(b'abcdef').hexdigest())
    assert destination.read_bytes() == b'abcdef'
    assert result['bytes'] == 6


def test_parallel_ranges_preserve_legacy_prefix(monkeypatch, tmp_path):
    """多连接恢复应保留旧单连接前缀，并按正确顺序拼接完整归档。"""
    payload = bytes(range(251)) * 13
    partial = tmp_path / 'archive.part'
    partial.write_bytes(payload[:512])

    def open_response(request, timeout):
        """独立按请求范围提供原始字节。"""
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        return Response(payload[start:end + 1], start, end, len(payload))

    monkeypatch.setattr(dataset_setup, 'urlopen', open_response)
