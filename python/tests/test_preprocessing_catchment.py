import os.path
from pathlib import Path

import hydrobricks as hb
from hydrobricks.preprocessing.catchment import Catchment

TEST_FILES_DIR = Path(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', 'tests', 'files', 'catchments'
)
CATCHMENT_OUTLINE = TEST_FILES_DIR / 'ch_sitter_appenzell' / 'outline.shp'
CATCHMENT_DEM = TEST_FILES_DIR / 'ch_sitter_appenzell' / 'dem.tif'


def has_required_packages() -> bool:
    return hb.has_rasterio and hb.has_geopandas and hb.has_shapely


def test_shapefile_parsing():
    if not has_required_packages():
        return
    Catchment(CATCHMENT_OUTLINE)


def test_dem_extraction():
    if not has_required_packages():
        return
    catchment = Catchment(CATCHMENT_OUTLINE)
    assert catchment.extract_dem(CATCHMENT_DEM)


def test_elevation_bands_isohypses():
    if not has_required_packages():
        return
    catchment = Catchment(CATCHMENT_OUTLINE)
    catchment.extract_dem(CATCHMENT_DEM)
    bands = catchment.get_elevation_bands(method='isohypse', distance=100)
    assert 74430000 < bands['area'].sum() < 74450000


def test_elevation_bands_quantiles():
    if not has_required_packages():
        return
    catchment = Catchment(CATCHMENT_OUTLINE)
    catchment.extract_dem(CATCHMENT_DEM)
    bands = catchment.get_elevation_bands(method='quantiles', number=25)
    assert 74430000 < bands['area'].sum() < 74450000
