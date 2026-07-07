# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io
import re
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from threadpoolctl import threadpool_limits

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.helper_functions import basic_statistics, median_table
from htc.utils.LabelMapping import LabelMapping
from htc.utils.LocationMap import LocationMap
from htc.utils.visualization import create_median_spectra_comparison_figure


class DatasetOverview:
    inner_threads = 8

    def __init__(self):
        self.width = 825
        self.files = {}

        # Match dataset scripts to datasets
        self.script_matches = {}
        for f in sorted((settings.htc_package_dir / "dataset_preparation").glob("run_dataset_*.py")):
            script = f.read_text()
            match = re.search(r'dataset_path = settings\.datasets\.([^[]+)\["path_dataset"\]', script)
            assert match is not None, f"Could not find the dataset_path line in {f}"

            self.script_matches[match.group(1)] = f

    def _introduction(self, name: str, df: pd.DataFrame) -> str:
        dataset_name = settings.datasets[name]["path_dataset"].name
        dsettings = DatasetSettings(settings.data_dirs[name])

        res = ""
        if "description" in dsettings:
            res += "\n\n".join(dsettings["description"]) + "\n\n"

        if "annotation_name_info" in dsettings:
            df = median_table(name, annotation_name="all")
            annotation_names = df["annotation_name"].unique()
            assert set(dsettings["annotation_name_info"].keys()) == set(annotation_names), (
                f"Annotation names do not match\nannotation names in the dataset: {annotation_names}\nannotation names described in the settings: {list(dsettings['annotation_name_info'].keys())}"
            )

            if "annotation_name_annotator" in dsettings:
                assert set(dsettings["annotation_name_annotator"].keys()).issubset(annotation_names), (
                    f"Annotator names for the annotation names are not a subset of the annotations in the dataset\nannotators in the dataset: {annotation_names}\nannotators described in the settings: {list(dsettings['annotation_name_annotator'].keys())}"
                )

            res += "<h3>Annotation Names</h3>\n"
            for annotation_name, info in dsettings["annotation_name_info"].items():
                if dsettings.get("annotation_name_default", "") == annotation_name:
                    res += f'- <b title="This is the primary annotation name which will be used if no explicit annotation name is specified.">`{annotation_name}`</b>: {info}'
                else:
                    res += f"- `{annotation_name}`: {info}"

                if (annotator := dsettings.get("annotation_name_annotator", {}).get(annotation_name)) is not None:
                    res += f' <span title="Main annotator">:fontawesome-solid-draw-polygon:</span> {annotator}'

                res += "\n"

            res += "\n"

        if "Tivita" in name:
            res += "This dataset contains images from the Tivita camera. For the general format of the files, please refer to the public [HeiPorSPECTRAL](https://spectralverse-heidelberg.org/HeiPorSPECTRAL) dataset which explains the meaning and format for many commonly used files.\n\n"

        if "ethics" in df.columns:
            res += f"This dataset contains information about ethics (reference numbers) per subject, see the `ethics` column in the metadata. The ethic reference numbers are usually listed in the paper (see existing papers for examples) and which numbers you need depends on the subjects you use in your analysis. Ethic documents are usually located in `$PATH_E130_Projekte/Biophotonics/Documents/Ethikantraege`. In total, this dataset contains animals from the following ethics: {', '.join(df['ethics'].unique())}.\n\n"

        res += f"""
<h3>General</h3>
- <span title="This class will be used when iterating over this dataset and hence contains information about the folder structure. It defines which information will be available as attributes when dealing with instances of this class.">DataPath</span> class: `{dsettings["data_path_class"]}`
- <span title="This script defines how the files of this dataset were generated.">Generation script</span>: `{self.script_matches[settings.datasets[dataset_name]["shortcut"]].name}`
- Shape of the images: {", ".join([str(x) for x in dsettings["shape"]])} ({", ".join(dsettings["shape_names"])})
"""

        return res

    def _access_info(self, name: str) -> str:
        # We need the name of the main dataset even for subdatasets
        dataset_name = settings.datasets[name]["path_dataset"].name

        res = ""

        if name != dataset_name:
            res += f"""
> &#x26a0;&#xfe0f; The dataset `{name}` is a subdataset of the dataset `{dataset_name}`. If you already have access to the main dataset, you can skip this section (it is identical to the main dataset).
"""

        res += f"""
<h3>Download the data</h3>
To download the data to your local htc folder, run the following command:
```bash
rsync -a --delete --info=progress2 --exclude=".git*" --exclude "data_original" $PATH_E130_Projekte/Biophotonics/Data/{dataset_name}/ ~/htc/{dataset_name}/
```
and add the line
```bash
export {settings.datasets[name]["env_name"]}=~/htc/{dataset_name}
```
to your `.env` file.
"""
        if (settings.datasets.network_data / dataset_name / ".git").exists():
            res += f"""
<h3>Upload the data</h3>
The data on the network drive is versioned via git. If you make changes to the dataset, please make sure the history is up-to-date. If you cloned the data repository to your local computer (or if you created the repository locally and pushed it to the network drive), you need to git commit and git push locally:
```bash
# Keep the history up-to-date
cd ~/htc/{dataset_name}
git status
git add .
git commit -m "Describe the changes to the dataset"
git push

# Sync only changes to the intermediates back:
rsync -a --info=progress2 --delete --exclude=".git*" ~/htc/{dataset_name}/intermediates/ $PATH_E130_Projekte/Biophotonics/Data/{dataset_name}/intermediates/
```

Otherwise, you can also sync all your changes back to the network drive and git commit directly in the folder on the network drive:
```bash
# Sync all changes back:
rsync -a --delete --info=progress2 --exclude=".git*" --exclude "data_original" ~/htc/{dataset_name}/ $PATH_E130_Projekte/Biophotonics/Data/{dataset_name}/

# Keep the history up-to-date on the network drive:
cd $PATH_E130_Projekte/Biophotonics/Data/{dataset_name}
git status
git add .
git commit -m "Describe the changes to the dataset"
```

In any case, it may also be wise to inform your colleagues about the changes you made (and maybe also update our GitLab runner).
"""
        else:
            res += f"""
<h3>Upload the data</h3>
If you make changes to the dataset, please make sure to sync them back to the network drive:
```bash
rsync -a --delete --info=progress2 --exclude=".git*" --exclude "data_original" ~/htc/{dataset_name}/ $PATH_E130_Projekte/Biophotonics/Data/{dataset_name}/
```
It may also be wise to inform your colleagues about the changes you made (and maybe also update our GitLab runner).
"""

        return res

    def _meta_info(self, name: str, df: pd.DataFrame) -> str:
        n_median_round = 4
        rows = []
        n_values_thresh = 10
        for c in sorted(df.columns):
            if c in ["path", "dataset_settings_path"]:
                continue

            try:
                # We use torch to select the lower value in case of even number of elements (numpy takes the average of the two values)
                x = df[c][~pd.isna(df[c])].to_numpy()
                x = torch.from_numpy(x)
                median = str(round(x.median().item(), n_median_round))
            except Exception:
                median = None

            try:
                dfc = df[c].value_counts(ascending=True)
                values = dfc.index.tolist()
                counts = dfc.tolist()
                if len(values) > 0:
                    n_nans = df[c].isna().sum()
                    current_row = {
                        "name": c,
                        "type": df[c].dtype.name,
                        "# unique": len(values),
                        f'<span title="Number of images where this meta information is missing (relative to the {len(df)} images in total)."># nans</span>': f"{n_nans} ({n_nans / len(df):.1f} %)",
                    }
                    if median is not None:
                        current_row[
                            f'<span title="Median value computed via torch.median() and rounded to maximal {n_median_round} decimal places.">median</span>'
                        ] = median
                    if len(values) <= n_values_thresh:
                        current_row[
                            f'<span title="List of values and corresponding (counts) if the number of unique values is ≤ {n_values_thresh}.">value counts</span>'
                        ] = ", ".join([f"{v} ({c})" for v, c in zip(values, counts, strict=True)])
                    rows.append(current_row)
            except Exception:
                pass

        dfm = pd.DataFrame(rows)
        dfm = dfm.convert_dtypes()

        # Sho all columns with a horizontal scrollbar
        with pd.option_context("display.max_columns", None, "display.width", None):
            table_output = repr(median_table(dataset_name=name))

        html = f"""
The table below lists all available meta data for this dataset. This information can either be accessed via `path.meta(NAME)` for each path:
```python
>>> from htc import DataPath
>>> path = DataPath.from_image_name("{df.iloc[0].image_name}")
>>> path.meta("Camera_CamID")
'{DataPath.from_image_name(df.iloc[0].image_name).meta("Camera_CamID")}'
```
or via the `median_table()` function for a selection of paths (or all paths from the dataset):
```python
>>> from htc import median_table
>>> median_table(dataset_name="{name}")
{table_output}
```
"""

        html += dfm.to_html(index=False, justify="left", border=0, escape=False).replace(' class="dataframe"', "")
        return html

    def _data_stats_figure(self, name: str) -> str:
        res = ""
        df = basic_statistics(name, annotation_name="all")
        for annotation_name in df["annotation_name"].unique():
            res += f"<h3>{annotation_name}</h3>"
            dfa = df[df["annotation_name"] == annotation_name]

            dfc = dfa.groupby("label_name", as_index=False)[["subject_name", "image_name"]].nunique()
            dfc = dfc.rename(
                columns={
                    "label_name": "label name",
                    "subject_name": '<span title="Number of unique subjects with this label."># subjects</span>',
                    "image_name": '<span title="Number of unique images with this label."># images</span>',
                }
            )
            res += dfc.to_html(index=False, justify="left", border=0, escape=False).replace(' class="dataframe"', "")

            labels = dfa["label_name"].unique().tolist()
            dfa = dfa.sort_values(by=["subject_name", "timestamp"])
            subjects = dfa["subject_name"].unique().tolist()
            counts = np.zeros((len(subjects), len(labels)), dtype=np.int64)

            for _, row in dfa.iterrows():
                if row["label_name"] in labels:
                    label_index = labels.index(row["label_name"])
                    subject_index = subjects.index(row["subject_name"])
                    counts[subject_index, label_index] += 1

            assert counts.shape[1] == len(labels) and counts.shape[0] == len(subjects), "Dimension mismatch"

            fig_labels = go.Figure()
            fig_labels.add_trace(
                go.Heatmap(
                    z=counts,
                    x=labels,
                    y=subjects,
                    text=counts,
                    texttemplate="%{text}",
                    colorscale="Teal",
                    colorbar=dict(title="<b># images</b>"),
                )
            )
            fig_labels.update_xaxes(title_text="<b>label name</b>")
            fig_labels.update_yaxes(title_text="<b>subject name</b>")
            fig_labels.update_layout(title_text="Number of images per label and subject", title_x=0.5)
            fig_labels.update_layout(width=self.width, height=len(subjects) * 20)
            fig_labels.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            res += fig_labels.to_html(full_html=False, include_plotlyjs="cdn")

            fig_pixels = px.box(dfa, x="label_name", y="n_pixels", category_orders={"label_name": labels})
            fig_pixels.update_traces(boxmean=True)
            fig_pixels.update_xaxes(title_text="<b>label name</b>")
            fig_pixels.update_yaxes(title_text="<b># pixels</b>")
            fig_pixels.update_layout(title_text="Distribution of annotated organ sizes", title_x=0.5)
            fig_pixels.update_layout(template="plotly_white")
            fig_pixels.update_layout(width=self.width)
            fig_pixels.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            res += fig_pixels.to_html(full_html=False, include_plotlyjs="cdn")

        return res

    def _location_map_figure(self, name: str) -> str:
        res = ""
        df = median_table(name, annotation_name="all")
        for annotation_name in df["annotation_name"].unique():
            res += f"<h3>{annotation_name}</h3>"
            dfa = df[df["annotation_name"] == annotation_name]

            paths = DataPath.from_table(dfa)

            # Label mapping which only includes the labels for the current annotation name
            labels = df[df["annotation_name"] == annotation_name]["label_name"].unique().tolist()
            mapping = LabelMapping(
                {l: i for i, l in enumerate(labels)}, label_colors=paths[0].dataset_settings.get("label_colors")
            )
            mapping.label_colors["unknown"] = "#ffffff"

            lm = LocationMap(paths, mapping)
            with threadpool_limits(self.inner_threads):
                lm.compute_map()
            fig = lm.create_figure()

            # The location maps can bloat up the website in total size so we use JPEG compression to keep the image file sizes reasonable
            f = io.BytesIO()
            fig.savefig(f, dpi=150, format="jpg", pil_kwargs=dict(optimize=True, quality=95))
            plt.close(fig)
            self.files[f"{name}/{annotation_name}_location_map.jpg"] = f.getvalue()

            res += f'<img alt="Location Map" src="{quote_plus(annotation_name)}_location_map.jpg">'

        return res

    def _median_spectra_figure(self, name) -> str:
        df = median_table(name, annotation_name="all")
        colors = {name: px.colors.qualitative.Alphabet[i] for i, name in enumerate(df["annotation_name"].unique())}

        fig = create_median_spectra_comparison_figure(df, group_column="annotation_name", color_mapping=colors)
        fig.update_layout(legend=dict(y=1, yref="container", yanchor="top", orientation="h"))
        fig.update_layout(width=self.width)
        fig.update_layout(title=None)

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def overview_page(self, dataset: dict[str, str | pd.DataFrame]) -> None:
        settings.log.info(f"Working on dataset {dataset['name']}")

        f = io.StringIO()
        print(f"# {dataset['name']}", file=f)

        print("## Introduction", file=f)
        print(self._introduction(dataset["name"], dataset["df_meta"]), file=f)

        print("## Data Access", file=f)
        print(self._access_info(dataset["name"]), file=f)

        print("## Meta Information", file=f)
        print(self._meta_info(dataset["name"], dataset["df_meta"]), file=f)

        if "subject_name" in dataset["df_meta"].columns:
            print("## Label Statistics", file=f)
            print("Basic information about the annotated regions stratified by annotation name.", file=f)
            print(self._data_stats_figure(dataset["name"]), file=f)

        if dataset["name"] != "2021_03_30_Tivita_studies":
            print("## Location Maps", file=f)
            print(
                "The location map shows a heatmap of the positions of the annotated pixels for each label and annotation name, i.e., it denotes where labels are usually located in an image. The exemplary RGB and segmentation images are representatives of their respective classes chosen to have a maximum overlap with the global location map. Interpretation of columns: LOCATION_HEATMAP EXAMPLE_SEGMENTATION, EXAMPLE_IMAGE",
                file=f,
            )
            print(self._location_map_figure(dataset["name"]), file=f)

            print("## Median Spectra", file=f)
            print(
                "Median spectra per label and annotation name. The shaded area denotes one standard deviation across subjects.",
                file=f,
            )
            print(self._median_spectra_figure(dataset["name"]), file=f)

        self.files[f"{dataset['name']}.md"] = f.getvalue()

    def save_files(self) -> None:
        # This import requires mkdorcs_gen_files to be generated which is done by the mkdocs build command within the docs dir (see .gitlab-ci.yml)
        import mkdocs_gen_files

        for path, data in self.files.items():
            with mkdocs_gen_files.open(path, "w" if type(data) == str else "wb") as f:
                f.write(data)


def generate_overview(dataset: dict[str, str | pd.DataFrame]) -> DatasetOverview:
    overview = DatasetOverview()
    overview.overview_page(dataset)

    return overview
