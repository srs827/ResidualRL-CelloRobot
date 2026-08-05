# Public Annotation Workflow

The annotator-facing workflow should not require coding, terminal commands, or
Label Studio accounts. Use the public browser packages by default.

## What Annotators Receive

Send each annotator one ZIP file from:

`SoundClassifier/Data_Collection/public_annotation_packages_a_final/`

- `A1_annotation_package.zip`
- `A2_annotation_package.zip`
- `A3_annotation_package.zip`
- `A4_annotation_package.zip`
- `A5_annotation_package.zip`

Each package contains:

- `index.html`: the browser annotation app
- `audio/`: that annotator's assigned WAV files
- `README.txt`: simple annotator instructions

Each annotator has 200 recordings. Every recording is assigned to two different
annotators.

## Instructions To Send Annotators

Copy/paste this:

```text
Thank you for helping annotate cello recordings.

1. Download the ZIP file I sent you.
2. Unzip it.
3. Open the unzipped folder.
4. Double-click index.html.
5. Listen to each recording and fill in the ratings.
6. Your progress saves automatically in that browser.
7. When finished, click Download Results.
8. Send me the downloaded annotations JSON file.

Please use a laptop/desktop browser if possible: Chrome, Safari, Edge, or
Firefox are all fine. Headphones are recommended.
```

## Annotation Form

Required technical-quality ratings use:

- 1: poor
- 2: fair
- 3: good
- 4: great

Required technical metrics:

- Overall quality
- Tone quality
- Bow control
- Attack quality
- Release quality
- Dynamic accuracy

Required consistency metrics:

- Dynamic consistency: gets much softer, gets softer, equal level throughout,
  gets louder, gets much louder
- Tone quality consistency: much clearer at start, clearer at start,
  consistent, clearer at finish, much clearer at finish
- Surface-noise consistency: much noisier at start, noisier at start, even
  noise level, noisier at finish, much noisier at finish

Optional tonal-character metrics:

- Grainy to smooth
- Harsh to sweet
- Thin to rich
- Airy to clear
- Dry to resonant

## Build Public Packages

From the repo root:

```bash
python3 SoundClassifier/Data_Collection/package_public_annotation.py \
  --tasks-dir SoundClassifier/Data_Collection/label_studio_tasks_a_final \
  --audio-dir SoundClassifier/Data_Collection/dataset_a_final/audio \
  --out-dir SoundClassifier/Data_Collection/public_annotation_packages_a_final \
  --annotators A1 A2 A3 A4 A5
```

Then create/send ZIPs:

```bash
cd SoundClassifier/Data_Collection/public_annotation_packages_a_final
python3 -m zipfile -c A1_annotation_package.zip A1
python3 -m zipfile -c A2_annotation_package.zip A2
python3 -m zipfile -c A3_annotation_package.zip A3
python3 -m zipfile -c A4_annotation_package.zip A4
python3 -m zipfile -c A5_annotation_package.zip A5
```

## Prepare Dataset And Task Splits

The completed recording set is `dataset_a_final`.

Deduplicate and validate it:

```bash
python3 SoundClassifier/Data_Collection/prepare_annotation_dataset.py \
  SoundClassifier/Data_Collection/dataset_a_final
```

Create the initial four balanced task files:

```bash
cd SoundClassifier/Data_Collection
python3 label_studio_bridge.py export \
  --meta dataset_a_final/metadata.jsonl \
  --audio-dir dataset_a_final/audio \
  --document-root "$PWD" \
  --out-dir label_studio_tasks_a_final \
  --annotators A1 A2 A3 A4 \
  --annotations-per-sample 2 \
  --hide-condition
```

Then redistribute to five annotators while preserving A2's first 200 tasks and
dropping A2's original last 50. The current `label_studio_tasks_a_final` folder
already reflects this redistribution.

Expected assignment balance:

- A1: 200 recordings
- A2: 200 recordings
- A3: 200 recordings
- A4: 200 recordings
- A5: 200 recordings

Expected shared pair counts:

- Every annotator pair shares 50 recordings.

## Merge Returned Results

Put the returned files somewhere like:

`SoundClassifier/Data_Collection/returned_annotations/`

Then merge:

```bash
cd SoundClassifier/Data_Collection
python3 label_studio_bridge.py import \
  --meta dataset_a_final/metadata.jsonl \
  --exports \
    A1=returned_annotations/annotations_A1.json \
    A2=returned_annotations/annotations_A2.json \
    A3=returned_annotations/annotations_A3.json \
    A4=returned_annotations/annotations_A4.json \
    A5=returned_annotations/annotations_A5.json
```

The public browser app exports Label-Studio-shaped JSON, so the existing import
path works unchanged.

## Internal Alternatives

Label Studio and the terminal annotator remain available for internal use, but
they are not the public annotator workflow.

Start Label Studio:

```bash
./SoundClassifier/Data_Collection/start_label_studio.sh
```

Use terminal annotation with an assigned task file:

```bash
python3 SoundClassifier/Data_Collection/annotate.py \
  --meta SoundClassifier/Data_Collection/dataset_a_final/metadata.jsonl \
  --audio-dir SoundClassifier/Data_Collection/dataset_a_final/audio \
  --annotator A1 \
  --task-file SoundClassifier/Data_Collection/label_studio_tasks_a_final/tasks_A1.json \
  --hide-condition
```
