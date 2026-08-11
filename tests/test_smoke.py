import mosaic


def test_grid_default():
    assert mosaic.GRID_DEFAULT == (64, 64, 3)
