# Training Pipeline

Workflow to train the custom CNN on FER2013.

## Step-by-step

1. **Download FER2013** — requires Kaggle API token in `~/.kaggle/kaggle.json`:
   ```bash
   # PowerShell:
   cd training
   python -m pip install kaggle
   kaggle datasets download -d msambare/fer2013 -p ../data
   # Then unzip manually or use: Expand-Archive -Path "../data/fer2013.zip" -DestinationPath "../data"
   ```

2. **Test the data pipeline**:
   ```bash
   python data.py
   ```
   Should print shape info and class distribution.

3. **Train baseline (ANN) and CNN**:
   ```bash
   python train.py
   ```
   This runs ~80 epochs total (30 for ANN, 50 for CNN) unattended.
   - Saves models to `../artifacts/ann_baseline.keras` and `../artifacts/cnn.keras`
   - Saves confusion matrices and metrics JSON
   - Total runtime: 1–3 hours depending on CPU

4. **Parity-check** (verify Python predictions before export):
   ```bash
   python parity_check.py
   ```
   Generates `../artifacts/parity_check.json` with 10 test predictions.

5. **Export to TF.js**:
   ```bash
   python export_tfjs.py
   ```
   Converts `cnn.keras` to web-loadable format in `../web/model/cnn/`.

6. **Test in browser**:
   - Start the web server: `cd ../web && python -m http.server 8000`
   - Open http://localhost:8000
   - In browser console:
     ```javascript
     // Load the custom CNN and swap it in
     const modelUrl = 'model/cnn/model.json';
     const model = await tf.loadLayersModel(modelUrl);
     window.moodMirror.setClassifier({
       name: 'Custom CNN',
       classify(face) { ... }
     });
     ```

## Files

- `data.py`: Load FER2013, augmentation, CLAHE preprocessing
- `models.py`: ANN baseline and CNN architectures
- `train.py`: Main training loop with evaluation and metrics
- `export_tfjs.py`: Keras → TensorFlow.js converter
- `parity_check.py`: Validate Python vs. TF.js prediction parity

## Expected results

After training (Section 8 of PRD):

| Metric | Expected range |
|---|---|
| Majority-class baseline | ~25% (always predict happy) |
| ANN test accuracy | ~50–55% |
| CNN test accuracy | ~60–65% |
| CNN improvement over ANN | +8–12 percentage points |

FER2013 is genuinely hard (human agreement is ~65%), so don't expect 90%+.
