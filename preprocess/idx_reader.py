import gzip
import os
from ipaddress import IPv4Address

import numpy as np

SESSION_BYTE_LEN = 13


class DataLoader:
    """ Load data """

    def _load(self, file_path, **kwargs):
        """ Should be overriden by sub-classes """
        raise NotImplementedError()

    def load(self, file_path, **kwargs):
        if not os.path.exists(file_path):
            raise AssertionError('File {} not exists'.format(file_path))
        return self._load(file_path, **kwargs)


class IdxFileLoader(DataLoader):
    """ Load idx format data """

    def _read_bytes(self, byte_stream, num_byte):
        """ read n bytes from byte stream into a big endian integer """
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(byte_stream.read(num_byte), dtype=dt)[0]

    def _load(self, file_path, **kwargs):
        # need this fileld
        assert 'gzip_compressed' in kwargs
        gzip_compressed = kwargs['gzip_compressed']
        assert gzip_compressed in [True, False]

        # according to the value of gzip_compressed, utilize different stream
        with gzip.open(file_path) if gzip_compressed is True else open(file_path, 'rb') as f:
            # read magic number and do assertion
            magic_numbers = f.read(4)
            assert magic_numbers[0] == 0 and magic_numbers[1] == 0
            data_type = magic_numbers[2]
            if data_type != 8:
                raise AssertionError('Only support for unsigned char now')

            # extract shape
            shape = magic_numbers[3]

            # extract number of samples
            num_samples = self._read_bytes(f, 4)

            # extract dimensions
            dimensions = np.array([], dtype=np.int32)
            for _ in range(shape - 1):
                dimensions = np.append(dimensions, self._read_bytes(f, 4))

            # read data
            buf = f.read(num_samples * np.prod(dimensions, dtype=np.int32))

            # assert this is the end of data
            assert f.read() == b''

            # read from buffer
            data = np.frombuffer(buf, dtype=np.uint8)

            # assert enough data are read
            assert len(data.shape) == 1 and data.shape[0] == num_samples * np.prod(dimensions, dtype=np.int32)

            # reshape data
            data = data.reshape(np.append(num_samples, dimensions))
            return data_type, shape, dimensions, num_samples, data


def _big_endian_bytes2int(byte_data):
    res = 0
    power = 1
    for each_byte_data in byte_data[::-1]:
        res += power * each_byte_data
        power *= 256
    return res


def _read(dimensions, stream):
    if len(dimensions) == 0:
        return ord(stream.read(1))
    elif len(dimensions) == 1:
        return [val for val in stream.read(dimensions[0])]
    else:
        res = []
        for _ in range(dimensions[0]):
            res.append(_read(dimensions[1:], stream))
        return res


def _extract_session_info(byte_data):
    assert len(byte_data) == SESSION_BYTE_LEN
    assert byte_data[0] == 0 or byte_data[0] == 1
    is_tcp = byte_data[0] == 1
    ip0 = str(IPv4Address(_big_endian_bytes2int(byte_data[1:5])))
    port0 = _big_endian_bytes2int(byte_data[5:7])
    ip1 = str(IPv4Address(_big_endian_bytes2int(byte_data[7:11])))
    port1 = _big_endian_bytes2int(byte_data[11:13])
    return {
        'protocol': 'TCP' if is_tcp else 'UDP',
        'ip0': ip0,
        'port0': port0,
        'ip1': ip1,
        'port1': port1,
    }


def read_images(idx_filename, feature_type='ip-above', only_images=True):
    """
    Extract information image from idx file

    Parameters
    ----------
    idx_filename: str
    feature_type: str
        could be 'ip-above' or 'payload-len' for different type of image feature file
    only_images: bool
        only return images or
        with session information, actual packet count if feature_type is 'ip-above'
        with session information, actual packet count, actual byte count if feature_type is 'payload-len'

    Returns
    -------
    `numpy.ndarray`
        1. images if only_images is True
        2.1 (session information, actual packet count, images) if feature_type is 'ip-above'
        2.2 (session information, actual packet count, actual byte count, images) if feature_type is 'payload-len'
    """
    with gzip.open(idx_filename, 'rb') as f:
        magic_numbers = f.read(4)
        # print('magic_number', magic_numbers)
        assert magic_numbers[0] == 0 and magic_numbers[1] == 0
        if magic_numbers[2] != 8:
            raise AssertionError('Only support for unsigned char')
        shape = magic_numbers[3]
        # print('shape', shape)
        num_examples = int.from_bytes(f.read(4), byteorder='big')
        # print('number of examples',num_examples)
        dimensions = []
        for _ in range(shape - 1):
            dimensions.append(int.from_bytes(f.read(4), byteorder='big'))
        # print('dimensions', dimensions)
        data_list = []
        for _ in range(num_examples):
            each_data_point = _read(dimensions, f)
            session_info = _extract_session_info(each_data_point[:SESSION_BYTE_LEN])
            if feature_type == 'payload-len':
                actual_pkt_count = int.from_bytes(each_data_point[SESSION_BYTE_LEN:SESSION_BYTE_LEN+4], byteorder='big')
                actual_byte_count = int.from_bytes(each_data_point[SESSION_BYTE_LEN + 4:SESSION_BYTE_LEN + 6], byteorder='big')
                other_data = each_data_point[SESSION_BYTE_LEN + 6:]
            elif feature_type == 'ip-above':
                actual_pkt_count = each_data_point[SESSION_BYTE_LEN]
                other_data = each_data_point[SESSION_BYTE_LEN + 1:]
            else:
                raise AssertionError('feature_type could only be "payload-len" or "ip-above"')
            if only_images is True:
                data_list.append(other_data)
            elif only_images is False:
                if feature_type == 'ip-above':
                    data_list.append((session_info, actual_pkt_count, other_data))
                elif feature_type == 'payload-len':
                    data_list.append((session_info, actual_pkt_count, actual_byte_count, other_data))
                else:
                    raise AssertionError('feature_type could only be "payload-len" or "ip-above"')
            else:
                raise AssertionError('only_images could only be True or False')

        assert f.read() == b''

    data_list = np.array(data_list)
    return data_list


def read_labels(idx_filename):
    """
    Extract labels from idx file

    Parameters
    ----------
    idx_filename: str

    Returns
    -------
    [int]
        labels
    """
    return IdxFileLoader().load(idx_filename, gzip_compressed=True)[-1]
