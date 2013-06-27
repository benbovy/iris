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
"""Rules for converting BPCH fields into cubes."""

import warnings

import numpy as np
import netcdftime

import iris
import iris.fileformats.manager
from iris.coords import DimCoord
from iris.exceptions import TranslationError


__all__ = ['run']

# time encoding for bpch files
TIME_UNIT = iris.unit.Unit('hours since 1985-01-01 00:00:00',
                           calendar=iris.unit.CALENDAR_STANDARD)

def name(cube, field):
    """Set the cube's name from the field."""
    cube.rename(field.cf_name)


def units(cube, field):
    """
    Set the cube's units from the field.
    Unhandled units are stored in an "invalid_units" attribute instead.
    """
    units = field.unit.strip()
    try:
        cube.units = units
    except ValueError:
        # Just add it as an attribute.
        warnings.warn("Unhandled units '{0}' recorded in cube attributes.".
                      format(units))
        cube.attributes["invalid_units"] = units


def time(cube, field):
    """Add a time coord to the cube."""
    point = field.times[0]
    bounds = field.times

    time_coord = DimCoord(points=point,
                          bounds=bounds,
                          standard_name='time',
                          units=TIME_UNIT)

    cube.add_aux_coord(time_coord)


def add_coords(cube, field, ctm_grid_coords):
    """Add horizontal/vertical coordinates to the cube, from the CTM grid"""
    
    x_coord = DimCoord(ctm_grid_coords['lon'], standard_name="grid_longitude",
                       units="degrees")
    y_coord = DimCoord(ctm_grid_coords['lat'], standard_name="grid_latitude",
                       units="degrees")
    
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    
    # add vertical coordinates only for 3D fields
    if len(field.shape) > 2:
        layers_coord = DimCoord(np.arange(1, field.shape[2] + 1),
                                    standard_name="model_level_number",
                                    units="1")
        cube.add_dim_coord(layers_coord, 2)
        
        if field.shape[2] == ctm_grid_coords['Nlayers']:
            press_coord = DimCoord(ctm_grid_coords['pressure'],
                                   standard_name="air_pressure", units="hPa")
            alt_coord = DimCoord(ctm_grid_coords['altitude'] * 1000.0,
                                 standard_name="altitude", units="m")
            if ctm_grid_coords['hybrid']:
                eta_coord = DimCoord(ctm_grid_coords['eta'],
                        standard_name="atmosphere_hybrid_height_coordinate",
                        units="1")
                sigma_coord = None
            else:
                eta_coord = None
                sigma_coord = DimCoord(ctm_grid_coords['sigma'],
                                standard_name="atmosphere_sigma_coordinate",
                                units="1")
            
            
            for c in (press_coord, alt_coord, eta_coord, sigma_coord):
                if c is not None:
                    cube.add_aux_coord(c, 2)


def attributes(cube, field):
    """Add attributes to the cube."""
    
    def add_attr(name):
        """Add an attribute to the cube."""
        if hasattr(field, name):
            value = getattr(field, name)
            cube.attributes[name] = value

    add_attr("name")
    add_attr("full_name")
    add_attr("index")
    add_attr("number")
    add_attr("category")
    add_attr("modelname")
    add_attr("chemical")
    add_attr("molecular_weight")
    add_attr("carbon_weight")
    add_attr("hydrocarbon")
    add_attr("scale")


def run(field, ctm_grid_coords):
    """
    Convert a BPCH field to an Iris cube.

    Args:

        * field - a :class:`~iris.fileformats.bpch.BPCHField`

    Returns:

        * A new :class:`~iris.cube.Cube`, created from the BPCHField.

    """
    # Figure out what the eventual data type will be after any scale
    # transforms.
    dummy_data = np.zeros(1, dtype='f')
    if hasattr(field, 'scale'):
        dummy_data = field.scale * dummy_data
    
    # Create cube with data (not yet deferred)
    data_proxy = np.array(field.data_proxy)
    data_manager = iris.fileformats.manager.DataManager(field.shape,
                                                        dummy_data.dtype,
                                                        None)
    
    cube = iris.cube.Cube(data_proxy, data_manager=data_manager)
    
    name(cube, field)
    units(cube, field)
    time(cube, field)
    add_coords(cube, field, ctm_grid_coords)
    attributes(cube, field)

    return cube
