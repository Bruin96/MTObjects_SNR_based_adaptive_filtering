"""Input/output functions."""

import sys
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.io import fits
from astropy.table import Table, Column, vstack, unique


def astro_deg_to_rad(angle):
    """Convert angles from astronomical format to radians."""
    return np.deg2rad(90+angle)


def standardise_table_headings(table, degtorad):
    """Standardise headings and angles of astro data tables."""

    for col_name in table.colnames:
        table[col_name].name = col_name.lower()

    if 'r' in table.colnames:
        table['r'].name = 'a'
    elif 'reff' in table.colnames:
        table['reff'].name = 'a'

    if 'arat' in table.colnames:
        table['arat'] = table['a']*table['arat']
        table['arat'].name = 'b'

    if 'pa' in table.colnames:
        table['pa'].name = 'theta'

    if 'theta' in table.colnames:
        if degtorad:
            table['theta'] = astro_deg_to_rad(table['theta'])

    return table


def get_file_extension(filename):
    """Get the extension part of a given filename."""

    if "." in filename:
        return filename.split(".")[-1]
    else:
        return None


def read_fits_image(filename, hdu_index=0):
    """Open a .fits file.
       Return the first data frame as a numpy array.
    """

    # Open the file
    try:
        hdulist = fits.open(filename)
        img_data = None

        while img_data is None:
            # Extract image data from file and close
            try:
                img_data = hdulist[hdu_index].data
                hdu_index += 1

            except IndexError:
                print("Could not find image data in file.")
                hdulist.close()
                sys.exit(1)

        hdulist.close()

        return img_data

    except IOError:
        print("Could not read file:", filename)
        sys.exit(1)


def write_fits_image(data, header=None, filename='out.fits'):
    """Create a new fits object from data and headers, and write to file."""
    # Create hdu objects
    primary_hdu = fits.PrimaryHDU(data[0])

    if header is not None:
        primary_hdu.header = header

    image_hdus = [fits.ImageHDU(d) for d in data[1:]]

    # Ignore clobber warning
    warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, append=True)

    # Write to file
    hdulist = fits.HDUList([primary_hdu, *image_hdus])
    hdulist.writeto(filename, clobber=True)  # Clobber deprecated for astropy 1.3
    hdulist.close()


def get_fits_header(filename):
    """Get the headers of the first frame of a .fits file"""
    try:
        # Open the file and read the header
        hdulist = fits.open(filename)
        header = hdulist[0].header
        hdulist.close()
        return header

    except IOError:
        print("Could not read file:", filename)
    except:
        print("Could not read header")
    finally:
        # Create an empty header if one could not be found
        header = fits.Header()

    return header


def read_data_table(filename, degtorad=False, standardise=True):
    """Read a file into an astropy table."""
    if standardise:
        return standardise_table_headings(Table.read(filename), degtorad)
    else:
        return Table.read(filename)


def read_multi_tables(tables, degtorad=True, standardise=True, id_init=0):
    """Read multiple table files into a single table with unique ids."""

    hybrid_table = read_data_table(tables[0], degtorad, standardise)

    for t in tables:
        this_table = read_data_table(t, degtorad, standardise)
        hybrid_table = vstack([hybrid_table, this_table])

    #hybrid_table = unique(hybrid_table)

    if 'id' not in hybrid_table.colnames:
        hybrid_table.add_column(Column(name='id', 
            data=np.arange(id_init, id_init + len(hybrid_table), dtype=int)))

    return hybrid_table


def save_csv(data, filename='out.csv', headings=''):
    """Save a table as a csv file."""
    np.savetxt(filename, data, header=headings)
