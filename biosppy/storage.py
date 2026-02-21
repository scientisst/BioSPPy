# -*- coding: utf-8 -*-
"""
biosppy.storage
---------------

This module provides several data storage methods.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range
import six

# built-in
import datetime
import json
import os
import zipfile
import struct

# 3rd party
import h5py
import numpy as np
import shortuuid
import joblib

# local
from . import utils


def serialize(data, path, compress=3):
    """Serialize data and save to a file using sklearn's joblib.

    Parameters
    ----------
    data : object
        Object to serialize.
    path : str
        Destination path.
    compress : int, optional
        Compression level; from 0 to 9 (highest compression).

    """

    # normalize path
    utils.normpath(path)

    joblib.dump(data, path, compress=compress)


def deserialize(path):
    """Deserialize data from a file using sklearn's joblib.

    Parameters
    ----------
    path : str
        Source path.

    Returns
    -------
    data : object
        Deserialized object.

    """

    # normalize path
    path = utils.normpath(path)

    return joblib.load(path)


def dumpJSON(data, path):
    """Save JSON data to a file.

    Parameters
    ----------
    data : dict
        The JSON data to dump.
    path : str
        Destination path.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'w') as fid:
        json.dump(data, fid)


def loadJSON(path):
    """Load JSON data from a file.

    Parameters
    ----------
    path : str
        Source path.

    Returns
    -------
    data : dict
        The loaded JSON data.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'r') as fid:
        return json.load(fid)


def zip_write(fid, files, recursive=True, root=None):
    """Write files to zip archive.

    Parameters
    ----------
    fid : file-like object
        The zip file to write into.
    files : iterable
        List of files or directories to pack.
    recursive : bool, optional
        If True, sub-directories and sub-folders are also written to the
        archive.
    root : str, optional
        Relative folder path.

    Notes
    -----
    * Ignores non-existent files and directories.

    """

    if root is None:
        root = ''

    for item in files:
        fpath = utils.normpath(item)

        if not os.path.exists(fpath):
            continue

        # relative archive name
        arcname = os.path.join(root, os.path.split(fpath)[1])

        # write
        fid.write(fpath, arcname)

        # recur
        if recursive and os.path.isdir(fpath):
            rfiles = [os.path.join(fpath, subitem)
                      for subitem in os.listdir(fpath)]
            zip_write(fid, rfiles, recursive=recursive, root=arcname)


def pack_zip(files, path, recursive=True, forceExt=True):
    """Pack files into a zip archive.

    Parameters
    ----------
    files : iterable
        List of files or directories to pack.
    path : str
        Destination path.
    recursive : bool, optional
        If True, sub-directories and sub-folders are also written to the
        archive.
    forceExt : bool, optional
        Append default extension.

    Returns
    -------
    zip_path : str
        Full path to created zip archive.

    """

    # normalize destination path
    zip_path = utils.normpath(path)

    if forceExt:
        zip_path += '.zip'

    # allowZip64 is True to allow files > 2 GB
    with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as fid:
        zip_write(fid, files, recursive=recursive)

    return zip_path


def unpack_zip(zip_path, path):
    """Unpack a zip archive.

    Parameters
    ----------
    zip_path : str
        Path to zip archive.
    path : str
        Destination path (directory).

    """

    # allowZip64 is True to allow files > 2 GB
    with zipfile.ZipFile(zip_path, 'r', allowZip64=True) as fid:
        fid.extractall(path)


def alloc_h5(path):
    """Prepare an HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path):
        pass


def store_h5(path, label, data):
    """Store data to HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.
    label : hashable
        Data label.
    data : array
        Data to store.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path) as fid:
        label = str(label)

        try:
            fid.create_dataset(label, data=data)
        except (RuntimeError, ValueError):
            # existing label, replace
            del fid[label]
            fid.create_dataset(label, data=data)


def load_h5(path, label):
    """Load data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.
    label : hashable
        Data label.

    Returns
    -------
    data : array
        Loaded data.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path) as fid:
        label = str(label)

        try:
            return fid[label][...]
        except KeyError:
            return None


def store_txt(path, data, sampling_rate=1000., resolution=None, date=None,
              labels=None, precision=6):
    """Store data to a simple text file.

    Parameters
    ----------
    path : str
        Path to file.
    data : array
        Data to store (up to 2 dimensions).
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    resolution : int, optional
        Sampling resolution.
    date : datetime, str, optional
        Datetime object, or an ISO 8601 formatted date-time string.
    labels : list, optional
        Labels for each column of `data`.
    precision : int, optional
        Precision for string conversion.

    Raises
    ------
    ValueError
        If the number of data dimensions is greater than 2.
    ValueError
        If the number of labels is inconsistent with the data.

    """

    # ensure numpy
    data = np.array(data)

    # check dimension
    if data.ndim > 2:
        raise ValueError("Number of data dimensions cannot be greater than 2.")

    # build header
    header = "Simple Text Format\n"
    header += "Sampling Rate (Hz):= %0.2f\n" % sampling_rate
    if resolution is not None:
        header += "Resolution:= %d\n" % resolution
    if date is not None:
        if isinstance(date, six.string_types):
            header += "Date:= %s\n" % date
        elif isinstance(date, datetime.datetime):
            header += "Date:= %s\n" % date.isoformat()
    else:
        ct = datetime.datetime.utcnow().isoformat()
        header += "Date:= %s\n" % ct

    # data type
    header += "Data Type:= %s\n" % data.dtype

    # labels
    if data.ndim == 1:
        ncols = 1
    elif data.ndim == 2:
        ncols = data.shape[1]

    if labels is None:
        labels = ['%d' % i for i in range(ncols)]
    elif len(labels) != ncols:
        raise ValueError("Inconsistent number of labels.")

    header += "Labels:= %s" % '\t'.join(labels)

    # normalize path
    path = utils.normpath(path)

    # data format
    p = '%d' % precision
    if np.issubdtype(data.dtype, np.integer):
        fmt = '%d'
    elif np.issubdtype(data.dtype, np.float):
        fmt = '%%.%sf' % p
    elif np.issubdtype(data.dtype, np.bool_):
        fmt = '%d'
    else:
        fmt = '%%.%se' % p

    # store
    np.savetxt(path, data, header=header, fmt=fmt, delimiter='\t')


def load_txt(path):
    """Load data from a text file.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    data : array
        Loaded data.
    mdata : dict
        Metadata.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'rb') as fid:
        lines = fid.readlines()

    # extract header
    mdata_tmp = {}
    fields = ['Sampling Rate', 'Resolution', 'Date', 'Data Type', 'Labels']
    values = []
    for item in lines:
        if b'#' in item:
            item = item.decode('utf-8')
            # parse comment
            for f in fields:
                if f in item:
                    mdata_tmp[f] = item.split(':= ')[1].strip()
                    fields.remove(f)
                    break
        else:
            values.append(item)

    # convert mdata
    mdata = {}
    df = '%Y-%m-%dT%H:%M:%S.%f'
    try:
        mdata['sampling_rate'] = float(mdata_tmp['Sampling Rate'])
    except KeyError:
        pass
    try:
        mdata['resolution'] = int(mdata_tmp['Resolution'])
    except KeyError:
        pass
    try:
        dtype = mdata_tmp['Data Type']
    except KeyError:
        dtype = None
    try:
        d = datetime.datetime.strptime(mdata_tmp['Date'], df)
        mdata['date'] = d
    except (KeyError, ValueError):
        pass
    try:
        labels = mdata_tmp['Labels'].split('\t')
        mdata['labels'] = labels
    except KeyError:
        pass

    # load array
    data = np.genfromtxt(values, dtype=dtype, delimiter=b'\t')

    return data, mdata


def load_edf(path):
    """Load data from an EDF+ (European Data Format) file.

    Parameters
    ----------
    path : str
        Path to the EDF file.

    Returns
    -------
    signals : array
        Array of signals read from the EDF file. Each column represents a signal.
    mdata : dict
        Metadata extracted from the EDF file, including:
        - version : str
        - patient_id : str
        - recording_id : str
        - start_date : str
        - start_time : str
        - header_bytes : str
        - reserved : str
        - num_data_records : int
        - duration_per_data_record : float
        - num_signals : int
        - labels : list of str
        - units : list of str
        - sampling_rates : list of int
        - physical_min : list of float
        - physical_max : list of float
        - digital_min : list of int
        - digital_max : list of int
        - annotations : list of tuples (onset, duration, annotation)

    Notes
    -----
    This function reads the EDF file header and data records, scales the signals
    into physical units, and parses the annotations according to the EDF+ specification.
    """

    def parse_annotations(data):
        annotations = []
        i = 0
        while i < len(data):
            if data[i] == 0:
                break
            onset = ''
            duration = ''
            while data[i] != 20:
                onset += chr(data[i])
                i += 1
            i += 1
            if data[i] == 21:
                i += 1
                while data[i] != 20:
                    duration += chr(data[i])
                    i += 1
                i += 1
            annotation = ''
            while data[i] != 0:
                if data[i] == 20:
                    # convert to string in HH:MM:SS format
                    onset = float(onset)  # seconds
                    onset = str(datetime.timedelta(seconds=onset))

                    duration = float(duration) if duration else 0
                    duration = str(datetime.timedelta(seconds=duration))

                    # remove leading and trailing white space
                    annotation = annotation.strip()
                    if annotation != '':
                        annotations.append((onset, duration, annotation))
                    annotation = ''
                    i += 1
                else:
                    annotation += chr(data[i])
                    i += 1
            i += 1
        return annotations

    with open(path, 'rb') as f:
        # Read the header
        header = f.read(256)

        # Extract fixed fields
        version = header[:8].decode('ascii').strip()
        patient_id = header[8:88].decode('ascii').strip()
        recording_id = header[88:168].decode('ascii').strip()
        start_date = header[168:176].decode('ascii').strip()
        start_time = header[176:184].decode('ascii').strip()
        header_bytes = header[184:192].decode('ascii').strip()
        reserved = header[192:236].decode('ascii').strip()
        num_data_records = int(header[236:244].decode('ascii').strip())
        duration_per_data_record = float(header[244:252].decode('ascii').strip())
        num_signals = int(header[252:256].decode('ascii').strip())

        # Read signal metadata
        labels = [f.read(16).decode('ascii').strip() for _ in range(num_signals)]
        transducer_types = [f.read(80).decode('ascii').strip() for _ in range(num_signals)]
        units = [f.read(8).decode('ascii').strip() for _ in range(num_signals)]
        physical_min = [float(f.read(8).decode('ascii').strip()) for _ in range(num_signals)]
        physical_max = [float(f.read(8).decode('ascii').strip()) for _ in range(num_signals)]
        digital_min = [int(f.read(8).decode('ascii').strip()) for _ in range(num_signals)]
        digital_max = [int(f.read(8).decode('ascii').strip()) for _ in range(num_signals)]
        prefiltering = [f.read(80).decode('ascii').strip() for _ in range(num_signals)]
        num_samples_per_data_record = [int(f.read(8).decode('ascii').strip()) for _ in range(num_signals)]
        reserved_space = f.read(32 * num_signals).decode('ascii').strip()
        sampling_rates = [int(num_samples / duration_per_data_record) for num_samples in num_samples_per_data_record]

        # Read data records
        signals = [[] for _ in range(num_signals)]
        annotations = []
        for _ in range(num_data_records):
            for i in range(num_signals):
                num_samples = num_samples_per_data_record[i]
                if labels[i] == 'EDF Annotations':
                    annotation_data = f.read(num_samples * 2)
                    annotations.extend(parse_annotations(annotation_data))
                else:
                    for _ in range(num_samples):
                        signals[i].append(struct.unpack('<h', f.read(2))[0])

        # Scale the signals into physical units
        for i in range(num_signals-1):
            signals[i] = np.array(signals[i])
            signals[i] = (signals[i] - digital_min[i]) / (digital_max[i] - digital_min[i]) * (physical_max[i] - physical_min[i]) + physical_min[i]

        # remove annotation from signals
        if 'EDF Annotations' in labels:
            num_signals -= 1
            signals = signals[:-1]
            labels = labels[:-1]
            units = units[:-1]
            sampling_rates = sampling_rates[:-1]
            physical_min = physical_min[:-1]
            physical_max = physical_max[:-1]
            digital_min = digital_min[:-1]
            digital_max = digital_max[:-1]

        mdata = {
            'version': version,
            'patient_id': patient_id,
            'recording_id': recording_id,
            'start_date': start_date,
            'start_time': start_time,
            'header_bytes': header_bytes,
            'reserved': reserved,
            'num_data_records': num_data_records,
            'duration_per_data_record': duration_per_data_record,
            'num_signals': num_signals,
            'labels': labels,
            'units': units,
            'sampling_rates': sampling_rates,
            'physical_min': physical_min,
            'physical_max': physical_max,
            'digital_min': digital_min,
            'digital_max': digital_max,
            'annotations': annotations
        }

        return np.array(signals).T, mdata


class HDF(object):
    """Wrapper class to operate on BioSPPy HDF5 files.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    mode : str, optional
        File mode; one of:

        * 'a': read/write, creates file if it does not exist;
        * 'r+': read/write, file must exist;
        * 'r': read only, file must exist;
        * 'w': create file, truncate if it already exists;
        * 'w-': create file, fails if it already esists.

    """

    def __init__(self, path=None, mode='a'):
        # normalize path
        path = utils.normpath(path)

        # open file
        self._file = h5py.File(path, mode)

        # check BioSPPy structures
        try:
            self._signals = self._file['signals']
        except KeyError:
            if mode == 'r':
                raise IOError(
                    "Unable to create 'signals' group with current file mode.")
            self._signals = self._file.create_group('signals')

        try:
            self._events = self._file['events']
        except KeyError:
            if mode == 'r':
                raise IOError(
                    "Unable to create 'events' group with current file mode.")
            self._events = self._file.create_group('events')

    def __enter__(self):
        """Method for with statement."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Method for with statement."""

        self.close()

    def _join_group(self, *args):
        """Join group elements.

        Parameters
        ----------
        ``*args`` : list
            Group elements to join.

        Returns
        -------
        weg : str
            Joined group path.

        """

        bits = []
        for item in args:
            bits.extend(item.split('/'))

        # filter out blanks, slashes, white space
        out = []
        for item in bits:
            item = item.strip()
            if item == '':
                continue
            elif item == '/':
                continue
            out.append(item)

        weg = '/' + '/'.join(out)

        return weg

    def add_header(self, header=None):
        """Add header metadata.

        Parameters
        ----------
        header : dict
            Header metadata.

        """

        # check inputs
        if header is None:
            raise TypeError("Please specify the header information.")

        self._file.attrs['json'] = json.dumps(header)

    def get_header(self):
        """Retrieve header metadata.

        Returns
        -------
        header : dict
            Header metadata.

        """

        try:
            header = json.loads(self._file.attrs['json'])
        except KeyError:
            header = {}

        return utils.ReturnTuple((header,), ('header',))

    def add_signal(self,
                   signal=None,
                   mdata=None,
                   group='',
                   name=None,
                   compress=False):
        """Add a signal to the file.

        Parameters
        ----------
        signal : array
            Signal to add.
        mdata : dict, optional
            Signal metadata.
        group : str, optional
            Destination signal group.
        name : str, optional
            Name of the dataset to create.
        compress : bool, optional
            If True, the signal will be compressed with gzip.

        Returns
        -------
        group : str
            Destination group.
        name : str
            Name of the created signal dataset.

        """

        # check inputs
        if signal is None:
            raise TypeError("Please specify an input signal.")

        if mdata is None:
            mdata = {}

        if name is None:
            name = shortuuid.uuid()

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            # create group
            node = self._file.create_group(weg)

        # create dataset
        if compress:
            dset = node.create_dataset(name, data=signal, compression='gzip')
        else:
            dset = node.create_dataset(name, data=signal)

        # add metadata
        dset.attrs['json'] = json.dumps(mdata)

        # output
        grp = weg.replace('/signals', '')

        return utils.ReturnTuple((grp, name), ('group', 'name'))

    def _get_signal(self, group='', name=None):
        """Retrieve a signal dataset from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the signal dataset.

        Returns
        -------
        dataset : h5py.Dataset
            HDF5 dataset.

        """

        # check inputs
        if name is None:
            raise TypeError(
                "Please specify the name of the signal to retrieve.")

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        # get data
        try:
            dset = node[name]
        except KeyError:
            raise KeyError("Inexistent signal dataset.")

        return dset

    def get_signal(self, group='', name=None):
        """Retrieve a signal from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the signal dataset.

        Returns
        -------
        signal : array
            Retrieved signal.
        mdata : dict
            Signal metadata.

        Notes
        -----
        * Loads the entire signal data into memory.

        """

        dset = self._get_signal(group=group, name=name)
        signal = dset[...]

        try:
            mdata = json.loads(dset.attrs['json'])
        except KeyError:
            mdata = {}

        return utils.ReturnTuple((signal, mdata), ('signal', 'mdata'))

    def del_signal(self, group='', name=None):
        """Delete a signal from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the dataset.

        """

        dset = self._get_signal(group=group, name=name)

        try:
            del self._file[dset.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")

    def del_signal_group(self, group=''):
        """Delete all signals in a file group.

        Parameters
        ----------
        group : str, optional
            Signal group.

        """

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        if node.name == '/signals':
            # delete all elements
            for _, item in six.iteritems(node):
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError(
                        "Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError(
                    "Unable to delete object with current file mode.")

    def list_signals(self, group='', recursive=False):
        """List signals in the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        recursive : bool, optional
            If True, also lists signals in sub-groups.

        Returns
        -------
        signals : list
            List of (group, name) tuples of the found signals.

        """

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        out = []
        for name, item in six.iteritems(node):
            if isinstance(item, h5py.Dataset):
                out.append((group, name))
            elif recursive and isinstance(item, h5py.Group):
                aux = self._join_group(group, name)
                out.extend(self.list_signals(group=aux,
                                             recursive=True)['signals'])

        return utils.ReturnTuple((out,), ('signals',))

    def add_event(self,
                  ts=None,
                  values=None,
                  mdata=None,
                  group='',
                  name=None,
                  compress=False):
        """Add an event to the file.

        Parameters
        ----------
        ts : array
            Array of time stamps.
        values : array, optional
            Array with data for each time stamp.
        mdata : dict, optional
            Event metadata.
        group : str, optional
            Destination event group.
        name : str, optional
            Name of the dataset to create.
        compress : bool, optional
            If True, the data will be compressed with gzip.

        Returns
        -------
        group : str
            Destination group.
        name : str
            Name of the created event dataset.

        """

        # check inputs
        if ts is None:
            raise TypeError("Please specify an input array of time stamps.")

        if values is None:
            values = []

        if mdata is None:
            mdata = {}

        if name is None:
            name = shortuuid.uuid()

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            # create group
            node = self._file.create_group(weg)

        # create new event group
        evt_node = node.create_group(name)

        # create datasets
        if compress:
            _ = evt_node.create_dataset('ts', data=ts, compression='gzip')
            _ = evt_node.create_dataset('values',
                                        data=values,
                                        compression='gzip')
        else:
            _ = evt_node.create_dataset('ts', data=ts)
            _ = evt_node.create_dataset('values', data=values)

        # add metadata
        evt_node.attrs['json'] = json.dumps(mdata)

        # output
        grp = weg.replace('/events', '')

        return utils.ReturnTuple((grp, name), ('group', 'name'))

    def _get_event(self, group='', name=None):
        """Retrieve event datasets from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        Returns
        -------
        event : h5py.Group
            HDF5 event group.
        ts : h5py.Dataset
            HDF5 time stamps dataset.
        values : h5py.Dataset
            HDF5 values dataset.

        """

        # check inputs
        if name is None:
            raise TypeError(
                "Please specify the name of the signal to retrieve.")

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        # event group
        try:
            evt_group = node[name]
        except KeyError:
            raise KeyError("Inexistent event dataset.")

        # get data
        try:
            ts = evt_group['ts']
        except KeyError:
            raise KeyError("Could not find expected time stamps structure.")

        try:
            values = evt_group['values']
        except KeyError:
            raise KeyError("Could not find expected values structure.")

        return evt_group, ts, values

    def get_event(self, group='', name=None):
        """Retrieve an event from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        Returns
        -------
        ts : array
            Array of time stamps.
        values : array
            Array with data for each time stamp.
        mdata : dict
            Event metadata.

        Notes
        -----
        Loads the entire event data into memory.

        """

        evt_group, dset_ts, dset_values = self._get_event(group=group,
                                                          name=name)
        ts = dset_ts[...]
        values = dset_values[...]

        try:
            mdata = json.loads(evt_group.attrs['json'])
        except KeyError:
            mdata = {}

        return utils.ReturnTuple((ts, values, mdata),
                                 ('ts', 'values', 'mdata'))

    def del_event(self, group='', name=None):
        """Delete an event from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        """

        evt_group, _, _ = self._get_event(group=group, name=name)

        try:
            del self._file[evt_group.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")

    def del_event_group(self, group=''):
        """Delete all events in a file group.

        Parameters
        ----------
        group  str, optional
            Event group.

        """

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        if node.name == '/events':
            # delete all elements
            for _, item in six.iteritems(node):
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError(
                        "Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError(
                    "Unable to delete object with current file mode.")

    def list_events(self, group='', recursive=False):
        """List events in the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        recursive : bool, optional
            If True, also lists events in sub-groups.

        Returns
        -------
        events : list
            List of (group, name) tuples of the found events.

        """

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        out = []
        for name, item in six.iteritems(node):
            if isinstance(item, h5py.Group):
                try:
                    _ = item.attrs['json']
                except KeyError:
                    # normal group
                    if recursive:
                        aux = self._join_group(group, name)
                        out.extend(self.list_events(group=aux,
                                                    recursive=True)['events'])
                else:
                    # event group
                    out.append((group, name))

        return utils.ReturnTuple((out,), ('events',))

    def close(self):
        """Close file descriptor."""

        # flush buffers
        self._file.flush()

        # close
        self._file.close()


def _read_mesh_vertices(filepath):
    """
    Auxiliary function to read the data of a Biosense Webster .mesh file
    Returns the vertices, triangles, and data.
    
    Parameters
    ----------
    filepath : str
        Path to the .mesh file.
        
    Returns
    -------
    vertices : pandas.DataFrame
        DataFrame containing the vertices information.
    triangles : pandas.DataFrame
        DataFrame containing the triangles information.
    data : pandas.DataFrame
        DataFrame containing the vertices colors information.
        
    """
    vertices = []
    triangles = []
    data = []
    section = "empty"
    
    import pandas as pd

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Start after VerticesSection header
            if line == "[VerticesSection]":
                section = "vertices"
                next(f)  # skip comment line
            
            if line == "[TrianglesSection]":
                section = "triangles"
                next(f)  # skip comment line
            
            if line == "[VerticesColorsSection]":
                section = "data"
                next(f)  # skip comment line
                next(f)  # skip another comment line
            
            if line == "[VerticesAttributesSection]":
                break  # End of relevant sections
            
            if section == "vertices":
                values = line.split()
                temp = [float(v) for v in values[2:]]
                if len(temp) > 0:
                    vertices.append([float(v) for v in values[2:]])
                
            elif section == "triangles":
                values = line.split()
                temp = [float(v) for v in values[2:]]
                if len(temp) > 0:
                    triangles.append(temp)
                
            elif section == "data":
                values = line.split()
                temp = [float(v) for v in values[2:]]
                if len(temp) > 0:
                    data.append(temp)

    columns_vertices = [
        "X", "Y", "Z",
        "NormalX", "NormalY", "NormalZ",
        "GroupID"
    ]
    columns_triangles = ["Vertex0", "Vertex1", "Vertex2", "NormalX", "NormalY", "NormalZ", "GroupID"]
    columns_data = ["Unipolar", "Bipolar", "LAT", "Impedance", "A1", "A2", "A2-A1", "SCI", "ICL", "ACL", "Force", "Paso", "µBi"]
    return pd.DataFrame(vertices, columns=columns_vertices), pd.DataFrame(triangles, columns=columns_triangles), pd.DataFrame(data, columns=columns_data)

def _read_ecg_files(ECG_file):
    """ Auxiliary function to read the data of a Biosense Webster ECG_Export.txt file.
    Returns the channel names, reference channel, and voltages.
    
    Parameters
    ----------
    ECG_file : str
        Path to the ECG_Export.txt file.
        
    Returns
    ----------
    channel_names : list
        List of channel names.
    reference_channel : str
        Name of the reference channel.
    voltages : numpy.ndarray
        Signals of each channel.
        
    """
    if ECG_file is not None:
        with open(ECG_file, 'r') as f:
            data = f.readlines()
            line1 = data[0].strip().split(',')
            line2 = data[1].strip().split(',')
            line3 = data[2].strip().split(',')
            
            lib_chars = ('m1', 'wc')

            if not line3[:2] in lib_chars:
                # another version of the ecg file
                # try the next line
                reference_channel = data[2].split('=')[-1].split('\n')[0]
                line3 = data[3].strip().split(',')
                
        # check if line1 matches string
        if line1[0] != 'ECG_Export_4.0':
            print(f"Warning: Unexpected file format in {ECG_file}")
            
        gain = float(line2[0].split('= ')[1])
        
        channel_names = line3[0].split()
        
        voltages = data[4:]
        
        for i in range(len(voltages)):
            voltages[i] = voltages[i].split()
        
        voltages = np.array(voltages, dtype=float) * gain  # apply gain
        
        return channel_names, reference_channel, voltages


def _find_file(directory, start_str=None, end_str=None):
    """ Auxiliary function to find a file in a directory that starts with start_str and ends with end_str. 
    If start_str or end_str is None, it will be ignored in the search.
    
    Parameters
    ----------
    directory : str
        Directory to search in.
        start_str : str, optional
        String that the file should start with.
        end_str : str, optional
        String that the file should end with.
        
    Returns
    ----------
    str
        Path to the found file, or None if no file is found.
    """
    for file in os.listdir(directory):
        if start_str is not None and end_str is not None:
            if file.startswith(start_str) and file.endswith(end_str):
                return os.path.join(directory, file)
        elif start_str is not None:
            if file.startswith(start_str):
                return os.path.join(directory, file)
        elif end_str is not None:
            if file.endswith(end_str):
                return os.path.join(directory, file)
    return None


def _find_all_files(directory, start_str=None, end_str=None):
    """ Auxiliary function to find all files in a directory that start with start_str and end with end_str.
    If start_str or end_str is None, it will be ignored in the search.
    
    Parameters
    ----------
    directory : str
        Directory to search in.
    start_str : str, optional
        String that the file should start with.
    end_str : str, optional
        String that the file should end with.
        
    Returns
    ----------
    list of str
        List of paths to the found files, or an empty list if no file is found.
    """
    files_found = []
    for file in os.listdir(directory):
        if start_str is not None and end_str is not None:
            if file.startswith(start_str) and file.endswith(end_str):
                files_found.append(os.path.join(directory, file))
        elif start_str is not None:
            if file.startswith(start_str):
                files_found.append(os.path.join(directory, file))
        elif end_str is not None:
            if file.endswith(end_str):
                files_found.append(os.path.join(directory, file))
    return files_found


def load_carto_study(filename, verbose = 1): 
    """
    Loads a CARTO study from a .xml file. The function extracts the relevant information about the maps,
    points, and signals, and saves them as .csv files in a subfolder for each map. 
    
    Adapted to Python from original MATLAB code written by the OpenEP team [OpenEP]_ [OpenEP2]_.
    Available at: https://github.com/openep/openep-core
    
    Parameters:
    filename : str
        Path to the CARTO .xml file.
    
    References
    ----------
    .. [OpenEP] Williams SE and Linton NWF (Feb. 2026). OpenEP/openep-core: v1.0.03 (Version v1.0.03). Zenodo. https://doi.org/10.5281/zenodo.4471318
    .. [OpenEP2] Williams SE, Roney CH, Connolly A, Sim I, Whitaker J, O’Hare D, Kotadia I, O’Neill L, Corrado C, Bishop M, Niederer SA, Wright M, O’Neill M and Linton NWF (2021) OpenEP: A Cross-Platform Electroanatomic Mapping Data Format and Analysis Platform for Electrophysiology Research. Front. Physiol. 12:646023. doi: 10.3389/fphys.2021.646023

    """
    import pandas as pd
    import xml.etree.ElementTree as ET

    directory = filename.rsplit('\\', 1)[0]
    
    # load and parse the XML file

    tree = ET.parse(filename)

    map_branch = tree.find('Maps')

    nmaps = int(map_branch.attrib['Count'])
        
    for i in range(nmaps):
        
        map = map_branch.findall('Map')[i]
        map_name = map.attrib['Name']
        map_meshfile = map.attrib['FileNames']
        
        if verbose > 0:
        
            print(f"Loading map {i+1}/{nmaps}: {map_name}")
        
        n_points = int(map.findall('CartoPoints')[0].attrib['Count'])
        
        if int(n_points) > 0:
            ECG_file = _find_file(directory, map_name, "ECG_Export.txt")
            
            [channel_names, reference_channel, voltages] = _read_ecg_files(ECG_file)
            
            map_coordinates = np.zeros((n_points, 3))
            map_ids = np.zeros((n_points,), dtype=int)
            
            points_woi = np.zeros((n_points, 2))
            points_reference = np.zeros((n_points, 1))
            points_unipolar = np.zeros((n_points, 1))
            points_bipolar = np.zeros((n_points, 1))
            points_map_annotation = np.zeros((n_points, 1))
            
            signals = np.zeros((n_points, voltages.shape[0], voltages.shape[1]))
            
            points_filename = map_name + '_Points_Export.xml'
            all_points = ET.parse(os.path.join(directory, points_filename))
            
            for k in range(n_points):
                map_coordinates[k,:] = [float(j) for j in map.findall('CartoPoints')[0].findall('Point')[k].attrib['Position3D'].split()]
                map_ids[k] = int(map.findall('CartoPoints')[0].findall('Point')[k].attrib['Id'])
                
                point_filename = map_name + '_P' + str(map_ids[k]) + '_Point_Export.xml'
                point_tree = ET.parse(os.path.join(directory, point_filename))
                points_woi[k,0] = float(point_tree.find("WOI").attrib['From'])
                points_woi[k,1] = float(point_tree.find("WOI").attrib['To'])
                points_reference[k,0] = int(point_tree.find("Annotations").attrib['Reference_Annotation'])
                points_map_annotation[k,0] = int(point_tree.find("Annotations").attrib['Map_Annotation'])
                points_unipolar[k,0] = float(point_tree.find("Voltages").attrib['Unipolar'])
                points_bipolar[k,0] = float(point_tree.find("Voltages").attrib['Bipolar'])
                
                
                point_ecg_filename = (os.path.join(directory, map_name + '_P' + str(map_ids[k]) + '_ECG_Export.txt'))
                
                [channel_names, reference_channel, voltages] = _read_ecg_files(point_ecg_filename)
                
                for m in range(len(channel_names)):
                    try:
                        signals[k, :, m] = voltages[:, m]
                    except:
                        print(f"Warning: Could not load signal for point {map_ids[k]}, channel {channel_names[m]}")
                
                point_filename = all_points.findall('Point')[k].attrib['File_Name']
                point_filename = os.path.join(directory, point_filename)           
            
        # check if RF files exist
        RF_files = _find_all_files(directory, start_str='RF_'+map_name)
        
        contact_force = _find_all_files(directory, start_str='ContactForceInRF_'+map_name)

        for j in range(len(RF_files)):
            if j == 0:
                RF_df = pd.read_table(RF_files[j])
            else:
                RF_df = pd.concat([RF_df, pd.read_table(RF_files[j])])  
                
        for j in range(len(contact_force)):
            if j == 0:
                CF_df = pd.read_table(contact_force[j])
            else:
                CF_df = pd.concat([CF_df, pd.read_table(contact_force[j])])
                
        try:
            df_vertices, df_triangles, df_data = _read_mesh_vertices(os.path.join(directory,map_meshfile))
        except:
            print(f"Could not read mesh file: {map_meshfile} for map {map_name}")
            continue
        
        # create a subfolder for each map
        
        subfolder = os.path.join(directory, map_name)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        df_vertices.to_csv(os.path.join(subfolder, map_name+'_vertices.csv'), index=False)
        df_triangles.to_csv(os.path.join(subfolder, map_name+'_triangles.csv'), index=False)
        df_data.to_csv(os.path.join(subfolder, map_name+'_data.csv'), index=False)
        
        # save channel names (point 1 header) as csv
        channel_names_df = pd.DataFrame(channel_names, columns=['Channel_Name'])
        channel_names_df.to_csv(os.path.join(subfolder, map_name+'_channel_names.csv'), index=False)

        # save woi as csv
        points_woi = pd.DataFrame(points_woi, columns=['From', 'To'])
        points_woi.to_csv(os.path.join(subfolder, map_name+'_woi.csv'), index=False)
        
        # save reference annotations as csv
        points_reference = pd.DataFrame(points_reference, columns=['Reference_Annotation'])
        points_reference.to_csv(os.path.join(subfolder, map_name+'_reference_annotations.csv'), index=False)
        
        # save map annotations as csv
        points_map_annotation = pd.DataFrame(points_map_annotation, columns=['Map_Annotation'])
        points_map_annotation.to_csv(os.path.join(subfolder, map_name+'_map_annotations.csv'), index=False)
        
        # save point ids as csv
        points_df = pd.DataFrame(map_ids, columns=['Point_ID'])
        points_df.to_csv(os.path.join(subfolder, map_name+'_point_ids.csv'), index=False)
        
        # save map coordinates as csv
        map_coords_df = pd.DataFrame(map_coordinates, columns=['X','Y','Z'])
        map_coords_df.to_csv(os.path.join(subfolder, map_name+'_point_coords.csv'), index=False)

        # save unipolar and bipolar voltages as csv
        points_unipolar_df = pd.DataFrame(points_unipolar, columns=['Unipolar'])
        points_unipolar_df.to_csv(os.path.join(subfolder, map_name+'_unipolar.csv'), index=False)
        points_bipolar_df = pd.DataFrame(points_bipolar, columns=['Bipolar'])
        points_bipolar_df.to_csv(os.path.join(subfolder, map_name+'_bipolar.csv'), index=False)

        # new subfolder inside each map's directory for signals per point
        subsubfolder = os.path.join(subfolder, 'signals_per_point')
        if not os.path.exists(subsubfolder):
            os.makedirs(subsubfolder)
        for p in range(n_points):
            point_signals = pd.DataFrame(signals[p, :, :], columns=channel_names)
            point_signals.to_csv(os.path.join(subsubfolder, f'Point_{map_ids[p]}_signals.csv'), index=False)
        
        try: 
            CF_df.to_csv(os.path.join(subfolder, map_name+'_contact_force.csv'), index=False)
        except:
            pass
        try:    
            RF_df.to_csv(os.path.join(subfolder, map_name+'_rf_data.csv'), index=False)
        except:
            pass
        
    if verbose > 0:
        print("Done")

