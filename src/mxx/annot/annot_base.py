import os
import yaml
import warnings

class AnnotBase:
    def __init__(self, path_annot, logger):
        self._path_annot = path_annot
        self._logger = logger
        self._annot = {}
        self._load_annot()

    def _load_annot(self):
        if os.path.exists(self._path_annot):
            with open(self._path_annot, 'r') as f:
                self._annot = yaml.safe_load(f)
                if self._annot is None:
                    self._logger(f"[annot_base] empty file in {self._path_annot}")
                    self._annot = {}
                    self._save_annot()
        else:
            dir_annot = os.path.dirname(self._path_annot)
            if not os.path.exists(dir_annot):
                os.makedirs(dir_annot, exist_ok=True)
            self._save_annot()
            self._logger(f'[annot_base]: create annot file in {self._path_annot}')
    

    def _save_annot(self):
        with open(self._path_annot, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self._annot, 
                f, 
                allow_unicode=True, 
                default_flow_style=False,
                sort_keys=False 
            )

    def __contains__(self, idx):
        if self._annot is None:
            self._logger(f"[annot_base] empty file in {self._path_annot}")
        return idx in self._annot

    def get_annot(self, idx):
        if idx in self._annot:
            return self._annot[idx]
        return None

    def set_annot(self, key, value):
        self._annot[key] = value
        self._save_annot()
        return

    def rename_key(self, key, key_new):
        if key not in self._annot:
            self._logger.warning(f"[annot_base] [rename_key] key:{key} miss")
            return False
        item = self._annot[key]
        self._annot.pop(key, None)
        self._annot[key_new] = item 
        self._save_annot()
        return True

    def remove_key(self, key):
        if key not in self._annot:
            return
        self._annot.pop(key, None)
        self._save_annot()


    def check_key(self, key, value):
        if key not in self._annot:
            return False
        return self._annot[key] == value

    def overwrite_key(self, key, value_check, value_new):
        if key not in self._annot:
            return False
        if self._annot[key] == value_new:
            return True
        if value_check is None:
            self._annot[key] = value_new
            self._save_annot()
            self._logger(f"[annotation] [overwrite_key] img: {self._img.get_name()}, key: {key}, data_old: {self._annot[key]}, value_new: {value_new}")
            return True
        data_old = self._annot[key]
        if value_check in data_old:
            self._annot[key] = value_new
            self._save_annot()
            self._logger(f"[annotation] [overwrite_key] img: {self._img.get_name()}, key: {key}, data_old: {data_old}, value_new: {value_new}")
            return 

    def __contains__(self, key):
        return key in self._annot

    @property
    def annot(self):
        return self._annot
