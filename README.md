# S_index_Minijob
Scripts to combine S index catalog with Gaia data and rudimentary isochrone fitting (but unfinished)

The python scripts can be found in the script folder. I used jupyter notebooks but also included .py files containing all of the code of the notebooks.
The tables with catalogs and lists of stars that I created can be found in the tables folder.

This projects consists of 2 parts:

### Part 1: Combining the S index catalog with Gaia data
The catalog with S indizes of around 5000 stars from Saikia et al. (2018) is called *catalog_original.dat*. The script in *Prepare_database.ipynb* was used to download data from the Gaia website for each star, and then to combine this data with the catalog. The result is the table in *CombinedCatalog.csv*.

### Part 2: Trying to fit isochrones to estimate further parameters
The goal of this part was to estimate different stellar parameters by using the stellar evolutionary tracks or isochrones from *MIST* (http://waps.cfa.harvard.edu/MIST/model_grids.html). The code for this is in *Gaia_Isochrones_ExploreData.ipynb*.

At first I tried using the EEP tracks, and then extract parameters by comparing the position in the HR diagram (using the Teff and Luminosity values from Gaia). I used the project from https://gitlab.gwdg.de/m.dahlkemper/subgiants/-/blob/master/README.md as a base (that's why some functions have weird names or are not used after defining them...). The results I got from scipy.interpolate.interp2d weren't that great.

Then I wanted to directly use the magnitudes from Gaia instead of the Teff and L values, and compare them with the Mist isochrones (you need the *UBV(RI)c + 2MASS JHKs + Kepler + Hipparcos + Tycho + Gaia* isochrone set for this). This looke kind of promising, but in the color-color diagram the position of the isochrones do not align with the stars from the catalog. There is some kind of nonconstant offset and I was not able to figure out where that comes from.

I tried anyway to estimate parameters using a nearest-neighbour method, so for each star I adopt the parameters of the closest point of the isochrones. As a test I tried to recover the Teff and L values and compare them with the ones from Gaia. For Teff this worked okayish, but for L I was not contend with the result.
As I was researching how to improve this, I stumbled upon the *isochrones.py* package (not included in this repository, see https://isochrones.readthedocs.io/en/latest/install.html). I think this does basically what I wanted to achieve, but in a more complex and advanced way.

I suppose it makes sense to use *isochrones.py* instead of the methods I tried to implement, but I was not able to install this package on my machine and get it working properly yet.
