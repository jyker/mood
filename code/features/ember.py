'''Ember Features

Ember V2 features are extracted on 
`lief == 0.10.1`
`python == 3.6.8`
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler

from typing import List, Dict, Tuple


def min_max_scaler(x: float, x_min: float, x_max: float,
                   feature_range: Tuple[int, int]):
    x = float(x)
    dst_min, dst_max = feature_range
    # fix for estimate x_max, x_min
    if x > x_max:
        x_max = x
    if x < x_min:
        x_min = x

    x_std = (x - x_min) / (x_max - x_min)
    return x_std * (dst_max - dst_min) + dst_min


class FeatureType:
    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        '''Generate a feature vector from the raw features'''
        raise NotImplementedError

    def feature_names(self) -> List[str]:
        '''Get output feature names for transformation'''
        raise NotImplementedError


class ByteHistogram(FeatureType):
    name = 'histogram'
    dim = 256

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        values = np.array(data, dtype=np.float32)
        # original ember normalize
        if feature_range is None:
            sum = values.sum()
            normalized = values / sum
            return normalized
        # self-defined normalize
        else:
            scaler = MinMaxScaler(feature_range)
            return scaler.fit_transform([values])[0]

    def feature_names(self) -> List[str]:
        return ['histogram'] * 256


class ByteEntropyHistogram(FeatureType):
    name = 'byteentropy'
    dim = 256

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        values = np.array(data, dtype=np.float32)
        # original ember normalize
        if feature_range is None:
            sum = values.sum()
            normalized = values / sum
            return normalized
        # self-defined normalize
        else:
            scaler = MinMaxScaler(feature_range)
            return scaler.fit_transform([values])[0]

    def feature_names(self) -> List[str]:
        return ['byteentropy'] * 256


class StringExtractor(FeatureType):
    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj['strings']
        hist_divisor = float(
            data['printables']) if data['printables'] > 0 else 1.0
        if feature_range is None:
            return np.hstack([
                data['numstrings'], data['avlength'], data['printables'],
                np.asarray(data['printabledist']) / hist_divisor,
                data['entropy'], data['paths'], data['urls'], data['registry'],
                data['MZ']
            ]).astype(np.float32)
        else:
            return np.hstack([
                min_max_scaler(data['numstrings'], 0, 4096, feature_range),
                min_max_scaler(data['avlength'], 0, 64, feature_range),
                min_max_scaler(data['printables'], 0, 65535, feature_range),
                MinMaxScaler(feature_range).fit_transform(
                    [data['printabledist']])[0],
                min_max_scaler(data['entropy'], 0, 128, feature_range),
                min_max_scaler(data['paths'], 0, 128, feature_range),
                min_max_scaler(data['urls'], 0, 128, feature_range),
                min_max_scaler(data['registry'], 0, 128, feature_range),
                min_max_scaler(data['MZ'], 0, 64, feature_range)
            ]).astype(np.float32)

    def feature_names(self) -> List[str]:
        sub_names = [
            'numstrings', 'avlength', 'printables', 'printabledist', 'entropy',
            'paths', 'urls', 'registery', 'MZ'
        ]
        name_dims = [1, 1, 1, 96, 1, 1, 1, 1, 1]
        res = []
        for i, j in zip(sub_names, name_dims):
            res.extend([f'{self.name}:{i}'] * j)
        assert len(res) == self.dim
        return res


class GeneralFileInfo(FeatureType):
    name = 'general'
    dim = 10

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj['general']
        if feature_range is None:
            return np.asarray([
                data['size'], data['vsize'], data['has_debug'],
                data['exports'], data['imports'], data['has_relocations'],
                data['has_resources'], data['has_signature'], data['has_tls'],
                data['symbols']
            ]).astype(np.float32)
        else:
            return np.asanyarray([
                min_max_scaler(data['size'], 0, 819200, feature_range),
                min_max_scaler(data['vsize'], 0, 819200, feature_range),
                min_max_scaler(data['has_debug'], 0, 128, feature_range),
                min_max_scaler(data['exports'], 0, 128, feature_range),
                min_max_scaler(data['imports'], 0, 1024, feature_range),
                min_max_scaler(data['has_relocations'], 0, 128, feature_range),
                min_max_scaler(data['has_resources'], 0, 128, feature_range),
                min_max_scaler(data['has_signature'], 0, 128, feature_range),
                min_max_scaler(data['has_tls'], 0, 128, feature_range),
                min_max_scaler(data['symbols'], 0, 128, feature_range)
            ]).astype(np.float32)

    def feature_names(self) -> List[str]:
        sub_names = [
            'size', 'vsize', 'has_debug', 'exports', 'imports',
            'has_relocations', 'has_resources', 'has_signature', 'has_tls',
            'symbols'
        ]
        res = [f'{self.name}:{i}' for i in sub_names]
        assert len(res) == self.dim
        return res


class HeaderFileInfo(FeatureType):
    name = 'header'
    dim = 62

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        if feature_range is None:
            return np.hstack([
                data['coff']['timestamp'],
                FeatureHasher(10, input_type="string").transform(
                    [[data['coff']['machine']]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform(
                    [data['coff']['characteristics']]).toarray()[0],
                FeatureHasher(10, input_type="string").transform(
                    [[data['optional']['subsystem']]]).toarray()[0],
                FeatureHasher(10, input_type="string").transform(
                    [data['optional']['dll_characteristics']]).toarray()[0],
                FeatureHasher(10, input_type="string").transform(
                    [[data['optional']['magic']]]).toarray()[0],
                data['optional']['major_image_version'],
                data['optional']['minor_image_version'],
                data['optional']['major_linker_version'],
                data['optional']['minor_linker_version'],
                data['optional']['major_operating_system_version'],
                data['optional']['minor_operating_system_version'],
                data['optional']['major_subsystem_version'],
                data['optional']['minor_subsystem_version'],
                data['optional']['sizeof_code'],
                data['optional']['sizeof_headers'],
                data['optional']['sizeof_heap_commit'],
            ]).astype(np.float32)
        else:
            scaler = MinMaxScaler(feature_range)
            return np.hstack([
                min_max_scaler(data['coff']['timestamp'], 0,
                               pd.Timestamp('2023-01-01 00:00:00').value,
                               feature_range),
                scaler.fit_transform([
                    FeatureHasher(10, input_type="string").transform(
                        [[data['coff']['machine']]]).toarray()[0]
                ])[0],
                scaler.fit_transform([
                    FeatureHasher(10, input_type="string").transform(
                        [data['coff']['characteristics']]).toarray()[0]
                ])[0],
                scaler.fit_transform([
                    FeatureHasher(10, input_type="string").transform(
                        [[data['optional']['subsystem']]]).toarray()[0]
                ])[0],
                scaler.fit_transform([
                    FeatureHasher(10, input_type="string").transform(
                        [data['optional']['dll_characteristics']]).toarray()[0]
                ])[0],
                scaler.fit_transform([
                    FeatureHasher(10, input_type="string").transform(
                        [[data['optional']['magic']]]).toarray()[0]
                ])[0],
                min_max_scaler(data['optional']['major_image_version'], 0, 16,
                               feature_range),
                min_max_scaler(data['optional']['minor_image_version'], 0, 16,
                               feature_range),
                min_max_scaler(data['optional']['major_linker_version'], 0, 16,
                               feature_range),
                min_max_scaler(data['optional']['minor_linker_version'], 0, 16,
                               feature_range),
                min_max_scaler(
                    data['optional']['major_operating_system_version'], 0, 16,
                    feature_range),
                min_max_scaler(
                    data['optional']['minor_operating_system_version'], 0, 16,
                    feature_range),
                min_max_scaler(data['optional']['major_subsystem_version'], 0,
                               16, feature_range),
                min_max_scaler(data['optional']['minor_subsystem_version'], 0,
                               16, feature_range),
                min_max_scaler(data['optional']['sizeof_code'], 0, 819200,
                               feature_range),
                min_max_scaler(data['optional']['sizeof_headers'], 0, 10240,
                               feature_range),
                min_max_scaler(data['optional']['sizeof_heap_commit'], 0,
                               10240, feature_range)
            ]).astype(np.float32)

    def feature_names(self) -> List[str]:
        sub_names = [
            'coff:timestamp',
            'coff:machine',
            'coff:characteristics',
            'optional:subsystem',
            'optional:subsystem',
            'optional:dll_characteristics',
            'optional:magic',
            'optional:major_image_version',
            'optional:minor_image_version',
            'optional:major_linker_version',
            'optional:minor_linker_version',
            'optional:major_operating_system_version',
            'optional:minor_operating_system_version',
            'optional:major_subsystem_version',
            'optional:minor_subsystem_version',
            'optional:sizeof_code',
            'optional:sizeof_headers',
            'optional:sizeof_heap_commit',
        ]
        name_dims = [1] + [10] * 5 + [1] * 11
        res = []
        for i, j in zip(sub_names, name_dims):
            res.extend([f'{self.name}:{i}'] * j)
        assert len(res) == self.dim
        return res


class SectionInfo(FeatureType):
    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj['section']
        sections = data['sections']
        if feature_range is None:
            general = [
                len(sections),  # total number of sections
                # number of sections with nonzero size
                sum(1 for s in sections if s['size'] == 0),
                # number of sections with an empty name
                sum(1 for s in sections if s['name'] == ""),
                # number of RX
                sum(1 for s in sections
                    if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']
                    ),
                # number of W
                sum(1 for s in sections if 'MEM_WRITE' in s['props'])
            ]
            # gross characteristics of each section
            section_sizes = [(s['name'], s['size']) for s in sections]
            section_sizes_hashed = FeatureHasher(
                50, input_type="pair").transform([section_sizes]).toarray()[0]
            section_entropy = [(s['name'], s['entropy']) for s in sections]
            section_entropy_hashed = FeatureHasher(
                50,
                input_type="pair").transform([section_entropy]).toarray()[0]
            section_vsize = [(s['name'], s['vsize']) for s in sections]
            section_vsize_hashed = FeatureHasher(
                50, input_type="pair").transform([section_vsize]).toarray()[0]
            entry_name_hashed = FeatureHasher(50,
                                              input_type="string").transform(
                                                  [data['entry']]).toarray()[0]
            characteristics = [
                p for s in sections for p in s['props']
                if s['name'] == data['entry']
            ]
            characteristics_hashed = FeatureHasher(
                50,
                input_type="string").transform([characteristics]).toarray()[0]
        else:
            general = [
                # total number of sections
                min_max_scaler(len(sections), 0, 32, feature_range),
                # number of sections with nonzero size
                min_max_scaler(sum(1 for s in sections if s['size'] == 0), 0,
                               32, feature_range),
                # number of sections with an empty name
                min_max_scaler(sum(1 for s in sections if s['name'] == ""), 0,
                               32, feature_range),
                # number of RX
                min_max_scaler(
                    sum(1 for s in sections if 'MEM_READ' in s['props']
                        and 'MEM_EXECUTE' in s['props']), 0, 32,
                    feature_range),
                # number of W
                min_max_scaler(
                    sum(1 for s in sections if 'MEM_WRITE' in s['props']), 0,
                    32, feature_range)
            ]
            # gross characteristics of each section
            scaler = MinMaxScaler(feature_range)
            section_sizes = [(s['name'], s['size']) for s in sections]
            section_sizes_hashed = scaler.fit_transform([
                FeatureHasher(50, input_type="pair").transform([section_sizes
                                                                ]).toarray()[0]
            ])[0]
            section_entropy = [(s['name'], s['entropy']) for s in sections]
            section_entropy_hashed = scaler.fit_transform([
                FeatureHasher(50,
                              input_type="pair").transform([section_entropy
                                                            ]).toarray()[0]
            ])[0]
            section_vsize = [(s['name'], s['vsize']) for s in sections]
            section_vsize_hashed = scaler.fit_transform([
                FeatureHasher(50, input_type="pair").transform([section_vsize
                                                                ]).toarray()[0]
            ])[0]
            entry_name_hashed = scaler.fit_transform([
                FeatureHasher(50,
                              input_type="string").transform([data['entry']
                                                              ]).toarray()[0]
            ])[0]
            characteristics = [
                p for s in sections for p in s['props']
                if s['name'] == data['entry']
            ]
            characteristics_hashed = scaler.fit_transform([
                FeatureHasher(50,
                              input_type="string").transform([characteristics
                                                              ]).toarray()[0]
            ])[0]
        # return
        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed,
            section_vsize_hashed, entry_name_hashed, characteristics_hashed
        ]).astype(np.float32)

    def feature_names(self) -> List[str]:
        sub_names = [
            'num_sections',
            'num_section_nonzero_size',
            'num_section_empty_name',
            'num_section_read_execute',
            'num_section_write',
            'section_size',
            'section_entropy',
            'section_vsize',
            'entry_name',
            'characteristics',
        ]
        name_dims = [1] * 5 + [50] * 5
        res = []
        for i, j in zip(sub_names, name_dims):
            res.extend([f'{self.name}:{i}'] * j)
        assert len(res) == self.dim
        return res


class ImportsInfo(FeatureType):
    name = 'imports'
    dim = 1280

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        if feature_range is None:
            # unique libraries
            libraries = list(set([l.lower() for l in data.keys()]))
            libraries_hashed = FeatureHasher(
                256, input_type="string").transform([libraries]).toarray()[0]

            # A string like "kernel32.dll:CreateFileMappingA" for each imported function
            imports = [
                lib.lower() + ':' + e for lib, elist in data.items()
                for e in elist
            ]
            imports_hashed = FeatureHasher(
                1024, input_type="string").transform([imports]).toarray()[0]
        else:
            # unique libraries
            scaler = MinMaxScaler(feature_range)
            libraries = list(set([l.lower() for l in data.keys()]))
            libraries_hashed = scaler.fit_transform([
                FeatureHasher(256,
                              input_type="string").transform([libraries
                                                              ]).toarray()[0]
            ])[0]

            # A string like "kernel32.dll:CreateFileMappingA" for each imported function
            imports = [
                lib.lower() + ':' + e for lib, elist in data.items()
                for e in elist
            ]
            imports_hashed = scaler.fit_transform([
                FeatureHasher(1024,
                              input_type="string").transform([imports
                                                              ]).toarray()[0]
            ])[0]
        # return
        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)

    def feature_names(self):
        return ['imports:dll_name'] * 256 + ['imports:function'] * 1024


class ExportsInfo(FeatureType):
    name = 'exports'
    dim = 128

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        if feature_range is None:
            exports_hashed = FeatureHasher(128, input_type="string").transform(
                [data]).toarray()[0]
        else:
            scaler = MinMaxScaler(feature_range)
            exports_hashed = scaler.fit_transform([
                FeatureHasher(128,
                              input_type="string").transform([data
                                                              ]).toarray()[0]
            ])[0]
        return exports_hashed.astype(np.float32)

    def feature_names(self):
        return ['exports:function'] * 128


class DataDirectories(FeatureType):
    name = 'datadirectories'
    dim = 15 * 2
    name_order = [
        "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE",
        "CERTIFICATE_TABLE", "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE",
        "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE", "BOUND_IMPORT", "IAT",
        "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
    ]

    def transform(self, raw_obj: Dict, feature_range: Tuple[int, int]):
        data = raw_obj[self.name]
        features = np.zeros(2 * len(self.name_order), dtype=np.float32)
        if feature_range is None:
            for i in range(len(self.name_order)):
                if i < len(data):
                    features[2 * i] = data[i]["size"]
                    features[2 * i + 1] = data[i]["virtual_address"]
        else:
            for i in range(len(self.name_order)):
                if i < len(data):
                    features[2 * i] = min_max_scaler(data[i]["size"], 0,
                                                     819200, feature_range)
                    features[2 * i + 1] = min_max_scaler(
                        data[i]["virtual_address"], 0, 819200, feature_range)
        return features

    def feature_names(self):
        res = []
        for i in self.name_order:
            for j in ['size', 'virtual_address']:
                res.append(f'datadirectories:{i}:{j}')
        assert len(res) == self.dim
        return res


class EmberV2Feature:
    dim = 2381
    features: Dict[str, FeatureType] = {
        # 0-255 | 256
        'ByteHistogram': ByteHistogram(),
        # 256-511 | 256
        'ByteEntropyHistogram': ByteEntropyHistogram(),
        # 512-615 | 104
        'StringExtractor': StringExtractor(),
        # 616-625 | 10
        'GeneralFileInfo': GeneralFileInfo(),
        # 626-687 | 62
        'HeaderFileInfo': HeaderFileInfo(),
        # 688-942 | 255
        'SectionInfo': SectionInfo(),
        # 943-2222 | 1280
        'ImportsInfo': ImportsInfo(),
        # 2223-2350 | 128
        'ExportsInfo': ExportsInfo(),
        # 2351-2380  | 30
        'DataDirectories': DataDirectories(),
    }

    def __init__(self):
        feature_names = []
        for _, fe in self.features.items():
            feature_names.extend(fe.feature_names())

        self.idx2name = {i: j for i, j in enumerate(feature_names)}

    def transform(self,
                  raw_obj: Dict,
                  feature_range: Tuple[int, int] = None) -> np.array:
        """
        Parameters
        ----------
        raw_obj: Dict
            dict-like ember V2 feature
        feature_range : Tuple[min, max], optional
            Desired range of transformed data, by default None
        """
        feature_vectors = [
            fe.transform(raw_obj, feature_range)
            for fe in self.features.values()
        ]
        vector = np.hstack(feature_vectors).astype(np.float32)
        # fix nan and infinity
        np.nan_to_num(vector, copy=False)
        return vector