# Skin Cancer ML Model

Is a skin cancer classifier as good as its score says?

I'm a third-year CS student and I wanted a project on medical imaging that wasn't "I trained a CNN and got 99%". Lots of people have done that. So I took the standard model on a public skin lesion dataset and spent the project trying to find out how much of its score I could trust.

This is a study of how models get evaluated. It is not a diagnostic tool and shouldn't be used on any real skin lesion.

## The two questions

1. How much does a careless train/test split inflate the score?
2. Even with a fair split, how much of the score comes from the lesion, and how much from everything else in the photo?

## Data

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) has 10,015 dermoscopy photos in 7 classes (moles, melanoma, and five others). The catch is that the 10,015 photos only show 7,470 different lesions. 26% of lesions were photographed more than once, which means 45% of all photos have a sibling photo of the same lesion somewhere in the data.

About two thirds of the photos are harmless moles. A model that always answers "mole" gets about 67% accuracy without learning anything. Because of that I report **balanced accuracy** (every class counts the same) and macro-F1 instead of plain accuracy. Random guessing gives a balanced accuracy of 0.143.

The data isn't in this repo. Download it from Kaggle and put it in `data/archive-5/`.

## Setup

- ResNet-18 pretrained on ImageNet, fine-tuned for 5 epochs on 224x224 photos
- class-weighted loss, random flips, AdamW with learning rate 3e-4
- exactly the same settings for every experiment, only the data changes
- the test set is scored once at the end, I never picked the "best epoch" by looking at it
- **naive split:** random 80/20 split by photo
- **fair split:** 80/20 split by lesion, so all photos of one lesion end up on the same side

## Results

All numbers are from one training run (seed 0). Roughly 2,000 test photos each.

### 1. The naive split makes the model look better

With the naive split, 791 of the 2,003 test photos (39.5%) show a lesion that is also in the training set. With the fair split that number is 0.

| split | accuracy | balanced accuracy | macro-F1 |
|---|---|---|---|
| naive (by photo) | 0.761 | 0.790 | 0.669 |
| fair (by lesion) | 0.683 | 0.657 | 0.577 |

Balanced accuracy drops 13 points once the model can't have seen the same lesion before.

Melanoma recall went the other way (0.717 naive, 0.814 fair). I think that's noise. There are only a few hundred melanomas in the test set and it's a single run, and recall alone is easy to push up by predicting melanoma more often.

### 2. What is the model actually using?

Same fair split, but I change what the model is allowed to see. The mask is a circle in the middle of the photo. It covers half the picture and is filled with a flat grey.

| the model gets | balanced accuracy | macro-F1 |
|---|---|---|
| the full photo | 0.657 | 0.577 |
| only the circle in the middle | 0.555 | 0.424 |
| everything except the circle | 0.397 | 0.292 |
| no image at all, just age, sex and body location | 0.350 | 0.173 |
| random guessing | 0.143 | |

- With the middle of every photo hidden, the model still gets 0.397, almost three times random guessing. The skin around the lesion carries information about the diagnosis.
- Age, sex and body location with no image at all already give 0.350. A good part of the signal is about the patient, not about how the lesion looks.
- The middle alone gets 0.555, and adding the surroundings brings it to 0.657.
- I also made Grad-CAM heatmaps for the full-photo model. On 300 test photos, 70% of the model's attention falls inside the middle circle, which is 50% of the picture area. In the 8 photos I looked at by eye, the red sits on the lesion every time.

## What I can't claim

- I did not show that the model uses rulers, ink marks or other artefacts. Skin type, sun damage and how zoomed in the photo is also relate to the diagnosis for normal medical reasons. I didn't separate those from real artefacts. Checking whether the photos come from different clinics or cameras would be the next thing to try.
- The mask is a rough guess. The dataset has no outlines of the lesions, so I used a circle in the centre. Small lesions leave extra skin inside it, and big ones stick out past it, so the "everything except the circle" model may still see part of the lesion.
- The heatmaps can't separate "looks at the lesion" from "looks at the centre", because every lesion is photographed in the centre.
- One seed, one dataset, one architecture, five epochs. The gaps are big enough that I trust the direction, but I wouldn't trust the third decimal.
- The fair split groups by lesion and doesn't balance the classes, so the two test sets have slightly different class mixes.
- The age/sex/location baseline leaves out `dx_type` (how the diagnosis was confirmed), because that is only known after the diagnosis.

## Run it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order:

| notebook | what it does | time on an M-series Mac |
|---|---|---|
| `01_data_and_splits.ipynb` | looks at the data, makes the naive and fair splits, counts the leakage | a few minutes |
| `02_train_and_compare.ipynb` | trains on both splits and compares | about 30 min |
| `03_shortcut_tests.ipynb` | masking tests, the no-image baseline, heatmaps | about 45 min |

Each notebook starts with `SMOKE_TEST = True`, which runs a tiny version to check that everything works. Set it to `False` for the real run. The scores and figures end up in `results/`.

```
splits/     train and test CSVs from notebook 01
results/    scores and figures from notebooks 02 and 03
data/, cache/, .venv/    not in the repo, too big
```
