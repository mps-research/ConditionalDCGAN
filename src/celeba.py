from pathlib import Path
from PIL import Image


def create_dataset(image_dir, attr_file, label_conds, n_samples, out_dir):
    if not Path(out_dir).exists():
        Path(out_dir).mkdir()
    else:
        print(f'{out_dir} already exists.')
        return

    with open(attr_file) as f:
        f.readline()
        f.readline()
        n_images = [0, ] * len(label_conds)
        while True:
            line = f.readline()
            if not line:
                break
            tokens = [t for t in line.split(' ') if t]
            image_name = tokens[0]
            attr = [int(t) if t == '1' else 0 for t in tokens[1:]]

            for i, cond in enumerate(label_conds):
                if n_images[i] >= n_samples:
                    continue
                label_dir = Path(out_dir) / str(i)
                if not label_dir.exists():
                    label_dir.mkdir()
                if cond(attr):
                    Image.open(Path(image_dir) / image_name).save(label_dir / image_name)
                    n_images[i] += 1
                    break
