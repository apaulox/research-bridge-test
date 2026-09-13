"""Remove the unnecessary Transformers wrapper without changing EAT tensor computation."""
from pathlib import Path
import shutil
import os

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "submission_eat_xlsr_aug_resnet38_mean_v1"
NEW = ROOT / "submission_eat_xlsr_aug_resnet38_mean_v2"


def copy_asset(src, dst):
    if Path(src).stat().st_size > 10 * 1024**2:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def main():
    shutil.copytree(OLD, NEW, copy_function=copy_asset,
                    ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.pyc"))
    arch = NEW / "model/eat_fmc/eat_architecture"
    config = arch / "configuration_eat.py"
    source = config.read_text()
    source = source.replace("from transformers import PretrainedConfig", "import json\nfrom pathlib import Path")
    source = source.replace("class EATConfig(PretrainedConfig):", "class EATConfig:")
    source = source.replace('    model_type = "eat"', '''    model_type = "eat"

    @classmethod
    def from_pretrained(cls, path, local_files_only=True):
        # Only the bundled JSON is read; there is no remote model discovery.
        with (Path(path) / "config.json").open(encoding="utf-8") as handle:
            return cls(**json.load(handle))''')
    source = source.replace("        super().__init__(**kwargs)", "        for key, value in kwargs.items():\n            setattr(self, key, value)")
    config.write_text(source)
    model = arch / "modeling_eat.py"
    source = model.read_text().replace("from transformers import PreTrainedModel", "from torch import nn")
    source = source.replace("class EATModel(PreTrainedModel):", "class EATModel(nn.Module):")
    source = source.replace("        super().__init__(config)", "        super().__init__()\n        self.config = config")
    model.write_text(source)
    info = NEW / "model/MODEL_INFO.md"
    info.write_text(info.read_text() + "\nRuntime v2: EAT config reads local JSON and its wrapper is torch.nn.Module. No Transformers model-loading dependency. All model weights and tensor operations remain unchanged from v1. Server failure root cause awaits the actual error log.\n")
    print(NEW)


if __name__ == "__main__":
    main()
