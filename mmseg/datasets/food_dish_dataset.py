from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FoodDishDataset(BaseSegDataset):
    """Three-class dataset for dishes, leftovers, and background."""

    METAINFO = dict(
        classes=('dishes', 'leftovers', 'background'),
        palette=[[0, 255, 0], [128, 0, 255], [128, 128, 128]])

    def __init__(self, **kwargs):
        kwargs.setdefault('reduce_zero_label', False)
        super().__init__(**kwargs)
