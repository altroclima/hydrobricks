import os.path
import tempfile
from pathlib import Path

import hydrobricks.models as models


def test_socont_creation():
    model = models.Socont()
    assert model.name == 'Socont'


def test_socont_creation_with_soil_storage_nb():
    models.Socont(soil_storage_nb=2)


def test_socont_creation_with_solver():
    models.Socont(solver='Runge-Kutta')


def test_socont_creation_with_surface_runoff():
    models.Socont(surface_runoff='linear-storage')


def test_socont_creation_with_land_covers():
    cover_names = ['ground', 'aletsch_glacier']
    cover_types = ['ground', 'glacier']
    models.Socont(land_cover_names=cover_names, land_cover_types=cover_types)


def test_create_json_config_file_created():
    model = models.Socont(soil_storage_nb=2, surface_runoff='linear-storage')
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.create_config_file(tmp_dir, 'structure', 'json')
        assert os.path.isfile(Path(tmp_dir) / 'structure.json')


def test_create_json_config_file_content():
    cover_names = ['ground', 'aletsch_glacier']
    cover_types = ['ground', 'glacier']
    model = models.Socont(solver='Runge-Kutta', soil_storage_nb=2,
                          land_cover_names=cover_names,
                          land_cover_types=cover_types,
                          surface_runoff='linear-storage')
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.create_config_file(tmp_dir, 'structure', 'json')
        txt = Path(tmp_dir + '/structure.json').read_text()
        assert '"base": "Socont"' in txt
        assert '"solver": "Runge-Kutta"' in txt
        assert '"soil_storage_nb": 2' in txt
        assert '"surface_runoff": "linear-storage"' in txt
        assert 'ground' in txt
        assert 'aletsch_glacier' in txt


def test_create_yaml_config_file_created():
    model = models.Socont(soil_storage_nb=2, surface_runoff='linear-storage')
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.create_config_file(tmp_dir, 'structure', 'yaml')
        assert os.path.isfile(Path(tmp_dir) / 'structure.yaml')


def test_create_yaml_config_file_content():
    cover_names = ['ground', 'aletsch_glacier']
    cover_types = ['ground', 'glacier']
    model = models.Socont(solver='Runge-Kutta', soil_storage_nb=2,
                          land_cover_names=cover_names,
                          land_cover_types=cover_types,
                          surface_runoff='linear-storage')
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.create_config_file(tmp_dir, 'structure', 'yaml')
        txt = Path(tmp_dir + '/structure.yaml').read_text()
        assert 'base: Socont' in txt
        assert 'solver: Runge-Kutta' in txt
        assert 'soil_storage_nb: 2' in txt
        assert 'surface_runoff: linear-storage' in txt
        assert 'ground' in txt
        assert 'aletsch_glacier' in txt
