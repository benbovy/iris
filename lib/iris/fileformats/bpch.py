# (C) British Crown Copyright 2010 - 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Provides BPCH file format capabilities."""

import os
import glob
import struct

import numpy as np

import iris
from iris.fileformats import uff 
import iris.fileformats.bpch_load_rules


def read_fmt_string(text, fields):
    """
    Convert a string with a given format to a dictionary.
    """
    return dict((name, function(text[i:i + di].strip()))
                 for name, i, di, function in fields)

def read_fmt_file(filename, fields, skip='#', merge_fields=False):
    """
    Read a formatted text file.      
    """
    with open(filename, 'r') as textfile:
        data = (read_fmt_string(line, fields)
                for line in textfile if not line.startswith(skip))

        if merge_fields:
            fieldnames = (f[0] for f in fields)
            return dict(((k, tuple(d[k] for d in data)) for k in fieldnames))
        else:
            return tuple(data)


class Diagnostics(object):
    """
    Define all the diagnostics (and its metadata) that one can archive
    with GEOS-Chem. An instance of this class is commonly related to a couple
    of diaginfo.dat and tracerinfo.dat files.
    """

    diaginfo_format = (('offset', 0, 8, int),
                       ('category', 9, 40, str),
                       ('description', 49, 9999, str))

    tracerinfo_format = (('name', 0, 8, str),
                         ('full_name', 9, 30, str),
                         ('molecular_weight', 39, 10, float),
                         ('carbon_weight', 49, 3, int),
                         ('number', 52, 9, int),
                         ('scale', 61, 10, float),
                         ('unit', 72, 40, str))
    
    def __init__(self, diaginfo=None, tracerinfo=None):
        self.categories = dict()
        self.diagnostics = dict()
    
        if diaginfo is not None:
            self.import_diaginfo(diaginfo)
        if tracerinfo is not None:
            self.import_tracerinfo(tracerinfo)
    
    def import_diaginfo(self, filename, clear=True):
        """
        Read categories from the 'diaginfo.dat'-file with given `filename`.
        If `clear` is True, all existing categories are removed.
        """
        info = read_fmt_file(filename, self.diaginfo_format)
        if clear:
            self.categories.clear()
        self.categories.update(((d.pop('category'), d) for d in info))

    def import_tracerinfo(self, filename, clear=True):
        """
        Read diagnostics from the 'tracerinfo.dat'-file with given `filename`.
        If `clear` is True, all existing diagnostics are removed.
        """
        info = read_fmt_file(filename, self.tracerinfo_format)
        if clear:
            self.diagnostics.clear()
        self.diagnostics.update(((d['number'], d) for d in info))

        for k, v in self.diagnostics.items():
            if v['carbon_weight'] != 1:
                self.diagnostics[k]['hydrocarbon'] = True
                self.diagnostics[k]['molecular_weight'] = 12e-3    
                                          # molecular weight C atoms (kg/mole)
            else:
                self.diagnostics[k]['hydrocarbon'] = False

            self.diagnostics[k]['chemical'] = bool(v['molecular_weight'])


class BPCHDataProxy(object):
    """A reference to the data payload of a single field in a BPCH file."""

    __slots__ = ('path', 'variable_name', 'position', 'endian')

    def __init__(self, path, variable_name, position, endian):
        self.path = path
        self.variable_name = variable_name
        self.position = position
        self.endian = endian

    def __repr__(self):
        return '%s(%r, %r at %d)' % (self.__class__.__name__, self.path,
                               self.variable_name, self.position)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def __setstate__(self, state):
        for key, value in state.iteritems():
            setattr(self, key, value)

    def load(self, data_shape, data_type, mdi, deferred_slice):
        """
        Load the corresponding proxy data item and perform any deferred
        slicing.

        Args:

        * data_shape (tuple of int):
            The data shape of the proxy data item.
        * data_type (:class:`numpy.dtype`):
            The data type of the proxy data item.
        * mdi (float):
            The missing data indicator value.
        * deferred_slice (tuple):
            The deferred slice to be applied to the proxy data item.

        Returns:
            :class:`numpy.ndarray`

        """
        with uff.FortranFile(self.path, 'rb', self.endian) as ctm_file:
            ctm_file.seek(self.position)
            variable = np.array(ctm_file.readline('*f'))
            variable = variable.reshape(data_shape, order='F')
            payload = variable[deferred_slice]

        return payload


class BPCHField():
    """
    A data field from a BPCH file.

    Capable of converting itself into a :class:`~iris.cube.Cube`

    """
    def __init__(self, from_file=None, diagnostics=None):
        """
        Create a BPCHField object and optionally read from an open file.

        Example::

            with open("bpch_file", "rb") as infile:
                field = BPCHField(infile)

        """
        self.diagnostics = diagnostics
        
        if from_file is not None:
            self.file = from_file
            self.path = os.path.abspath(from_file.name)
            self.read(from_file)

    def read(self, ufffile):
        """Read the next field from the given file object."""
        self._read_header(ufffile)
        self._read_data(ufffile)
    
    def _read_header(self, ufffile):
        """Read first and second header line."""
        line = ufffile.readline('20sffii')
        modelname, res0, res1, halfpolar, center180 = line
        line = ufffile.readline('40si40sdd40s7i')
        category, index, unit, tau0, tau1, reserved = line[:6]
        dim0, dim1, dim2, dim3, dim4, dim5, skip = line[6:]
        
        self.index = int(index)
        self.category = str(category).strip()
        self.times = (tau0, tau1)
        self.modelname = str(modelname).strip()
        self.center180 = bool(center180)
        self.halfpolar = bool(halfpolar)
        self.origin = (dim3, dim4, dim5)
        self.resolution = (res0, res1)
        self.shape = (dim0, dim1, dim2)
        self.name = ""       # defined later from self.diagnostic
        
        self.unit = unit.strip()
        
        # verify consistency add attributes from the diagnostics instance
        if self.diagnostics is not None:
            
            if not isinstance(self.diagnostics, Diagnostics):
                raise ValueError("bad value for diagnostics argument")
            if self.category not in self.diagnostics.categories.keys():
                raise ValueError("bad diagnostic category '%s'"
                                 % self.category)
            offset = self.diagnostics.categories[self.category]['offset']
            self._number = offset + self.index
            if self._number not in self.diagnostics.diagnostics.keys():
                raise ValueError("bad tracer index %d and/or diagnostic"
                                 " category '%s'" %(self.index, self.category))

            diag_attr = [a[0] for a in Diagnostics.tracerinfo_format]
            if self.diagnostics is not None:
                diag_attr += [k for k in
                              self.diagnostics.diagnostics[self._number].keys()
                              if k not in diag_attr]
            for attr in diag_attr:
                setattr(self, attr,
                        self.diagnostics.diagnostics[self._number][attr])
        
        # CF-like variable name
        self.cf_name = "_".join((self.name.strip(), self.category.strip()))
    
    def _read_data(self, ufffile):
        """Read the data array (returns only a proxy)."""
        position = ufffile.tell()
        self.data_proxy = BPCHDataProxy(self.path, self.cf_name,
                                        position, self.file.endian)
        ufffile.skipline()
        

def load_cubes(filenames, callback=None, endian='>'):
    """
    Loads cubes from a list of BPCH filenames.

    Args:

    * filenames - list of BPCH filenames to load

    Kwargs:

    * callback - a function which can be passed on to
                 :func:`iris.io.run_callback`
    * endian - byte order of the binary file

    .. note::

        The resultant cubes may not be in the same order as in the files.

    """
    # Lazy import to avoid importing pygchem until
    # attempting to load a BPCH file (pygchem.grid module needed).
    import iris.fileformats.bpch_load_rules
    from pygchem import grid
    
    if isinstance(filenames, basestring):
        filenames = [filenames]

    for filename in filenames:
        
        # import metadata from tracerinfo.dat and diaginfo.dat files
        # that should be found in the same directory.
        dir_path = os.path.dirname(filename)
        diaginfo = os.path.join(dir_path, "diaginfo.dat")
        tracerinfo = os.path.join(dir_path, "tracerinfo.dat")
        if not os.path.exists(diaginfo) or not os.path.exists(tracerinfo):
            raise IOError("no 'diaginfo.dat' and/or 'tracerinfo.dat' found")
        diagnostics = Diagnostics(diaginfo, tracerinfo)
        
        for path in glob.glob(filename):
            with uff.FortranFile(filename, 'rb', endian) as ctm_file:

                filetype = ctm_file.readline().strip()
                fsize = os.path.getsize(filename)
                filetitle = ctm_file.readline().strip()
                ctm_grid_coords = dict()
                
                while ctm_file.tell() < fsize:
                    
                    field = BPCHField(ctm_file, diagnostics)
                    
                    # calculate grid coordinates once per file (assume all
                    # fields in the file use the same grid, except vertical). 
                    if not len(ctm_grid_coords.keys()):
                        
                        ctm_grid = grid.CTMGrid.from_model(field.modelname)
                        
                        lon, lat = ctm_grid.lonlat_centers
                        ctm_grid_coords['lon'] = lon
                        ctm_grid_coords['lat'] = lat
                        ctm_grid_coords['eta'] = ctm_grid.eta_centers
                        ctm_grid_coords['sigma'] = ctm_grid.sigma_centers
                        ctm_grid_coords['pressure'] = ctm_grid.pressure_centers
                        ctm_grid_coords['altitude'] = ctm_grid.altitude_centers
                        ctm_grid_coords['hybrid'] = ctm_grid.hybrid
                        ctm_grid_coords['Nlayers'] = ctm_grid.Nlayers
                    
                    cube = iris.fileformats.bpch_load_rules.run(field,
                                                                ctm_grid_coords)

                    # Were we given a callback?
                    if callback is not None:
                        cube = iris.io.run_callback(callback,
                                                    cube,
                                                    field,
                                                    filename)
                    if cube is None:
                        continue

                    yield cube


def load_cubes_le(filenames, callback=None):
    """
    Loads cubes from a list of BPCH filenames (little endian).

    Args:

    * filenames - list of BPCH filenames to load

    Kwargs:

    * callback - a function which can be passed on to
                 :func:`iris.io.run_callback`

    .. note::

        The resultant cubes may not be in the same order as in the files.

    """
    return load_cubes(filenames, callback, endian='<')
