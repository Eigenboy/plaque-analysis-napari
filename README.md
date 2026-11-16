# Plaque Analysis for Napari

Two Napari plugins for full-plate plaque images. **Agar Region** lets you
review and export the area to count. **Count Plaques** detects plaques within
that area and lets you edit the masks and export statistics.

## Install from GitHub

### 1. Set up Conda

If you don't have Conda, follow the [Conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
for your operating system. Open a new terminal after installing and check that
`conda --version` works.

### 2. Download the source code

If you have Git, clone the repository and enter the project folder.

```sh
git clone https://github.com/Eigenboy/plaque-analysis-napari.git
cd plaque-analysis-napari
```

You can also use **Code → Download ZIP** on the
[repository page](https://github.com/Eigenboy/plaque-analysis-napari).
Extract the ZIP and use `cd` to enter the extracted folder in your terminal.
For example, if you extracted it in Downloads on macOS, run
`cd ~/Downloads/plaque-analysis-napari-main`.

### 3. Create the environment and install both plugins

Run these commands from the project folder that contains `pyproject.toml`
and `count_plaques/`. The install command also installs Napari and its
required dependencies.

```sh
conda create -n plaque_analysis_napari python=3.13 pip
conda activate plaque_analysis_napari
python -m pip install -e . -e ./count_plaques
```

Keep the downloaded project folder in place. The editable installation uses
the source files in that folder.

### 4. Open Napari

With the environment active, run `napari`. Open **Agar Region (Plaque Analysis)**
or **Count Plaques (Plaque Counting)** from Napari's **Plugins** menu.
In a new terminal, activate the environment before launching Napari.

```sh
conda activate plaque_analysis_napari
napari
```

## Save the agar region

Choose **Plugins → Agar Region (Plaque Analysis)**. In the dock, use
**Open image…** to select a full-plate grayscale image. Press **Q** or use the
boundary button to show or hide the cyan agar outline.

Click **Edit boundary** to show the vertices. Drag a handle to move that part
of the outline. Napari’s **Insert vertex** and **Remove vertex** tools let you
add or remove handles. Use Napari’s **Pan/zoom** tool to return to navigation.

If automatic detection fails, click **Draw boundary**. Hold down the left
mouse button and trace the agar edge. Release to close the outline, then use
**Edit boundary** to adjust it or **Export agar mask** to save it.

**Export agar mask** saves a TIFF to the `out` subfolder of the image's folder.
The export follows your edited boundary. When automatic detection succeeds, an
export before any edits uses the detected mask. The mask has the same
dimensions as the source image. Pixels inside the counting region are 1 and
excluded pixels are 0. Napari displays the image
at its native bit depth with a contrast stretch for viewing.

The detector uses the ChemiDoc workflow's outer-dish fit, inner-rim fit, and
exclusion of very dark marks. It expects a complete round dish with a dark rim
on a brighter background.

## Count and edit plaques

Choose **Plugins → Count Plaques (Plaque Counting)**. Click **Open original
image…** and select the original plate image. The plugin looks for
`out/<image stem>_agar_mask.tif` in the image’s folder. Use **Choose agar mask…**
if your mask is saved elsewhere.

To resume a previous session, open the same original image and click
**Open saved plaque masks…**. The picker starts with the export at
`out/<image stem>_plaque_masks.tif` if it exists. Loading preserves the saved IDs
and areas. Split suggestions are generated from the saved labels, and Undo
starts a fresh history for this session.

Click **Count plaques** to run the detector using the saved mask. The detector places
its results in a separate **Plaque masks** Labels layer. Each plaque has a
positive integer ID and background is 0. The count is the number of distinct
positive IDs in the layer.

- Press **Q** or use the visibility button to hide or show the plaque masks.
  During split review, **Q** toggles the overlays so you can inspect the original
  image while staying on the same proposal. With both plugins open, **Q** follows
  the active dock or layer.
- Press **N** or click **New plaque**, then hold the left mouse button and trace the outline.
  Release to close and fill it with a new plaque ID. Keep drawing or press **N** again to stop.
- Use the Labels picker to select an existing ID, then Paint or Erase to change
  its area. The plugin enables **Preserve labels** to protect neighbouring IDs.
- Press **R** or click **Remove plaque**, then click inside the colored mask to
  delete the whole plaque. Keep clicking masks or press **R** again to stop.
- **Count plaques** also scans the plate and marks likely touching pairs in
  yellow. Click **Review splits** to zoom to each proposal. **Accept split** and
  **Skip** advance the queue. Press **A** to accept the current preview.
  **Stop split review** keeps the remaining suggestions.
  Use **Scan for splits** to search the current labels again once the queue is empty.
- Press **S** or click **Manual split**, then drag a divider across one merged mask
  from outside to outside. The two sides seed a watershed that preserves the
  entire original mask. Press **A** or click **Accept split** to apply the preview,
  or click **Cancel** to discard it.
  Draw another divider or press **S** again to stop.
- Press **U** or click **Undo** to reverse the last addition, removal, split, or Labels edit.
- **Export plaque masks** saves the edited integer labels as a TIFF in the
  imported image’s `out` folder.
- **Plaque statistics** uses the current edits in the Labels layer and saves
  three files in the same `out` folder. The CSV has one row per plaque ID with
  its area in pixel² and the major/minor axis ratio of its second-moment ellipse.
  Two PNG histograms show the area and aspect ratio distributions. Dashed lines
  mark the mean and median, and the legend gives the mean ± population standard
  deviation and the median. If the minor axis has zero length, the CSV leaves
  the aspect ratio blank and the aspect ratio histogram omits that value.

Counting reads the selected agar mask from disk. If you export a revised mask
to the same path, **Count plaques** uses the updated file. It prompts before
replacing the plaque layer.

## Reinstall the plugins

From the project folder, activate `plaque_analysis_napari` and run
`python -m pip install -e . -e ./count_plaques` again.
