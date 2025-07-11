import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import pytest
import tfs
from generic_parser import DotDict
from matplotlib.figure import Figure

from omc3.check_corrections import _get_measurement_filter, correction_test_entrypoint
from omc3.correction.constants import MODEL_MATCHED_FILENAME
from omc3.correction.handler import get_filename_from_parameter
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.definitions.optics import (
    FILE_COLUMN_MAPPING,
    RDT_COLUMN_MAPPING,
    TUNE_COLUMN,
    ColumnsAndLabels,
)
from omc3.model.constants import TWISS_DAT
from omc3.optics_measurements.constants import (
    EXT,
    MDL,
    NAME,
    NAME2,
    PHASE,
    PHASE_ADV,
    TUNE,
    S,
)
from omc3.plotting.plot_checked_corrections import (
    SINGLE_PLANE_FILES,
    SPLIT_ID,
    show_plots,
)
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from tests.accuracy.test_global_correction import get_normal_params, get_skew_params
from tests.conftest import ids_str


class TestFullRun:
    @pytest.mark.basic
    @pytest.mark.parametrize('orientation', ('skew', 'normal',))
    @pytest.mark.parametrize('use_filter', (True, False,), ids=ids_str("filter={}"))
    def test_lhc_corrections(self, tmp_path, model_inj_beams, orientation, use_filter):
        """ Checks that correction_test_entrypoint runs and that all the output
        data is there. Very simple test. """
        beam = model_inj_beams.beam
        correction_params = get_skew_params(beam) if orientation == 'skew' else get_normal_params(beam)
        _create_fake_measurement(tmp_path, model_inj_beams.model_dir, correction_params.twiss)
        n_meas_files = len(list(tmp_path.glob(f"*{EXT}")))

        output_dir = tmp_path / "Corrections"
        filter_kwargs = {}
        if use_filter:
            filter_kwargs = {
                'optics_params': correction_params.optics_params[:2],  # PHASE[XY] or F1001[IR]
                'modelcut': [0.01, 0.01],
                'errorcut': [0.01, 0.01],
            }
        optics_params = filter_kwargs.get("optics_params", [])

        # TEST RUN -----------------------------------------------------------------
        correction_test_entrypoint(
            # show=True,  # debugging
            meas_dir=tmp_path,
            output_dir=output_dir,
            corrections=[correction_params.correction_filename],
            plot=True,
            # filter args
            **filter_kwargs,
            # accelerator class params:
            **model_inj_beams,
        )
        # --------------------------------------------------------------------------

        # Check output -------------------------------------------------------------
        assert (output_dir / MODEL_MATCHED_FILENAME).is_file()

        tfs_files = list(output_dir.glob(f"*{EXT}"))
        twiss_files = list(output_dir.glob(f"twiss*{EXT}"))

        assert (len(tfs_files) - len(twiss_files)) == n_meas_files

        # remove kmod files from the list
        tfs_files = [tfs_file for tfs_file in tfs_files if "kmod" not in tfs_file.name]

        if use_filter:
            # test if this test will check for the filtered optics parameters: (a meta-test)
            assert any(self._file_in_optics_params(tfs_file, optics_params) for tfs_file in tfs_files)

        # Go through all the files and check their contents
        for tfs_file in tfs_files:
            is_masked = self._file_in_optics_params(tfs_file, optics_params)
            assert tfs_file.stat().st_size

            if tfs_file in twiss_files:
                continue

            # can read?
            df = tfs.read(tfs_file)

            # has longitudinal columns?
            assert df.columns.str.match(S).any()
            assert df.columns.str.match(f"{PHASE_ADV}.{MDL}").any()

            # Check tune in header
            for ntune in (1, 2):
                tune_map = TUNE_COLUMN.set_plane(ntune)
                assert len([k for k in df.headers.keys() if tune_map.column in k]) == 3
                assert tune_map.column in df.headers
                assert tune_map.diff_correction_column in df.headers
                assert tune_map.expected_column in df.headers

            # has the resulting columns? has expected headers?
            try:
                column_map = FILE_COLUMN_MAPPING[tfs_file.stem[:-1]]
            except KeyError:
                for column_map in RDT_COLUMN_MAPPING.values():
                    self._assert_all_check_colums(df, column_map)
                    self._check_rms_header(df, column_map, is_masked=is_masked)
            else:
                planed_map = column_map.set_plane(tfs_file.stem[-1].upper())
                self._assert_all_check_colums(df, planed_map)
                self._check_rms_header(df, planed_map, is_masked=is_masked)

        # Check plotting output
        pdf_files = list(output_dir.glob(f"*.{mpl.rcParams['savefig.format']}"))
        assert len(pdf_files) == (n_meas_files + 6) * 2  # rdts split into 4; plotting combined and individual
        for pdf_file in pdf_files:
            assert pdf_file.stat().st_size

    @staticmethod
    def _file_in_optics_params(tfs_file, optics_params):
        """" Check the optics-parameter is associated with the given file. """
        filenames = (get_filename_from_parameter(key, beta_filename="beta_amp") for key in optics_params)
        return tfs_file.name in filenames


    @staticmethod
    def _assert_all_check_colums(df, colmap: ColumnsAndLabels):
        for col in (colmap.column, colmap.expected_column, colmap.error_expected_column, colmap.error_delta_column, colmap.diff_correction_column):
            assert col in df.columns


    @staticmethod
    def _check_rms_header(df, colmap: ColumnsAndLabels, is_masked):
        assert colmap.expected_rms_header in df.headers
        assert colmap.delta_rms_header in df.headers
        if is_masked:
            assert colmap.expected_masked_rms_header in df.headers
            assert colmap.delta_masked_rms_header in df.headers
            assert df.headers[colmap.expected_masked_rms_header]
            assert df.headers[colmap.delta_masked_rms_header]
            assert df.headers[colmap.expected_masked_rms_header] != df.headers[colmap.expected_rms_header]
            assert df.headers[colmap.delta_masked_rms_header] != df.headers[colmap.delta_rms_header]


class TestMeasurementFilter:
    """ Some tests to check if the measurement filter does what we expect. """
    @pytest.mark.basic
    def test_no_filter(self):
        mask = _get_measurement_filter(tfs.TfsDataFrame(), DotDict(optics_params=None))
        assert isinstance(mask, dict)
        assert not mask

    @pytest.mark.basic
    @pytest.mark.parametrize('which_cut', ('model', 'error', 'modelerror'), ids=ids_str("cut={}"))
    @pytest.mark.parametrize('beta_filename', ('beta_amplitude', 'beta_phase', 'beta_kmod'))
    def test_filter_data(self, tmp_path, which_cut, beta_filename):
        for param in get_normal_params(beam=1).optics_params + get_skew_params(beam=1).optics_params:
            if param == TUNE:
                continue  # skip

            filename = get_filename_from_parameter(param, beta_filename=beta_filename)
            filestem = filename.replace(EXT, "")
            try:
                # Normal Columns
                column_map: ColumnsAndLabels = FILE_COLUMN_MAPPING[filestem[:-1]]
                column_map = column_map.set_plane(filestem[-1].upper())
            except KeyError:
                # RDT Columns
                letter_map = {col[0]: col for col in RDT_COLUMN_MAPPING.keys()}
                column_map: ColumnsAndLabels = RDT_COLUMN_MAPPING[letter_map[param[-1]]]

            filter_opt = DotDict(
                optics_params=[param],
                modelcut={param: 0.5 if 'model' in which_cut else None},
                errorcut={param: 0.15 if 'error' in which_cut else None},
                weights={param: 1.0},
                meas_dir=tmp_path,
                beta_filename=beta_filename,
            )

            names = ["A", "B", "C", "D"]
            measurement = tfs.TfsDataFrame(
                [[1, 0.1],
                 [1, 0.2],
                 [2, 0.1],
                 [1, 0.1]],
                index=names,
                columns=[column_map.column, column_map.error_column],
                headers = {"Q1": 1, "Q2": 2},
            )

            model = tfs.TfsDataFrame(
                [[1, ],
                 [1, ],
                 [1, ],
                 [1, ]],
                index=names,
                columns=[column_map.column,],
                headers={"Q1": 1, "Q2": 1},
            )

            measurement[column_map.delta_column] = measurement[column_map.column] - model[column_map.column]
            measurement[column_map.error_delta_column] = measurement[column_map.error_column]
            if column_map.column.startswith(PHASE):
                # NAME2 does not really matter, but needs to be present
                model[NAME2] = names
                measurement[NAME2] = names

            tfs.write(tmp_path / filename, measurement, save_index=NAME)
            mask = _get_measurement_filter(model, filter_opt)[filename]  # 'model' is only used for index intersection

            # Check names in mask
            assert "A" in mask
            if not column_map.column.startswith(PHASE):
                assert "D" in mask

            if "model" in which_cut:
                assert "C" not in mask
            else:
                assert "C" in mask

            if "error" in which_cut:
                assert "B" not in mask
            else:
                assert "B" in mask


class TestPlotting:

    @pytest.mark.basic
    def test_normal_params(self, monkeypatch):
        figure_dict = {
            f"correction1{SPLIT_ID}param1_x": Figure(),
            f"correction1{SPLIT_ID}param1_y": Figure(),
            f"correction1{SPLIT_ID}param_2_x": Figure(),
            f"correction1{SPLIT_ID}param_2_y": Figure(),
            f"correction2{SPLIT_ID}param1_x": Figure(),
            f"correction2{SPLIT_ID}param1_y": Figure(),
            f"correction2{SPLIT_ID}param_2_x": Figure(),
            f"correction2{SPLIT_ID}param_2_y": Figure(),
            "param1_x_PARAM1X": Figure(),
            "param1_y_PARAM1Y": Figure(),
            "param_2_x_PARAM2X": Figure(),
            "param_2_y_PARAM2Y": Figure(),
        }

        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.TabWidget", MockTabAndWindow)
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.PlotWidget", MockPlotWidget)
        window = MockTabAndWindow("")
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.VerticalTabWindow", window)

        show_plots(figure_dict)

        assert window.title != ""  # set to something else
        assert window.shown

        assert len(window.tabs) == 3  # 1 for all, 2 for correction[12]
        for idx_tab, tab in enumerate(window.tabs):
            if idx_tab:
                assert tab.title == f"correction{idx_tab}"

            assert len(tab.tabs) == 2  # two params

            for idx, plottab in enumerate(tab.tabs):
                assert isinstance(plottab, MockPlotWidget)
                if idx == 0:
                    assert plottab.title == "param1"
                else:
                    assert plottab.title == "param 2"

    @pytest.mark.basic
    def test_single_plane_files(self, monkeypatch):
        single_name = SINGLE_PLANE_FILES[0]  # so far only one anyway

        figure_dict = {
            f"correction1{SPLIT_ID}{single_name}x": Figure(),
            f"correction1{SPLIT_ID}param_2_x": Figure(),
            f"correction1{SPLIT_ID}param_2_y": Figure(),
            f"correction2{SPLIT_ID}{single_name}x": Figure(),
            f"correction2{SPLIT_ID}param_2_x": Figure(),
            f"correction2{SPLIT_ID}param_2_y": Figure(),
            f"{single_name}x_PARAM1X": Figure(),
            "param_2_x_PARAM2X": Figure(),
            "param_2_y_PARAM2Y": Figure(),
        }

        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.TabWidget", MockTabAndWindow)
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.PlotWidget", MockPlotWidget)
        window = MockTabAndWindow("")
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.VerticalTabWindow", window)

        show_plots(figure_dict)

        assert window.title != ""  # set to something else
        assert window.shown

        assert len(window.tabs) == 3  # 1 for all, 2 for correction[12]
        for idx_tab, tab in enumerate(window.tabs):
            if idx_tab:
                assert tab.title == f"correction{idx_tab}"

            assert len(tab.tabs) == 2  # two params

            for idx, plottab in enumerate(tab.tabs):
                assert isinstance(plottab, MockPlotWidget)
                if idx == 0:
                    assert plottab.title == single_name.replace("_"," ").strip()
                    assert isinstance(plottab.fig1, Figure)
                    assert plottab.fig2 is None
                else:
                    assert plottab.title == "param 2"
                    assert isinstance(plottab.fig1, Figure)
                    assert isinstance(plottab.fig2, Figure)

    @pytest.mark.basic
    def test_rdt_params(self, monkeypatch):
        figure_dict = {
            f"correction1{SPLIT_ID}f1001_amplitude": Figure(),
            f"correction1{SPLIT_ID}f1001_phase": Figure(),
            f"correction1{SPLIT_ID}f1001_real": Figure(),
            f"correction1{SPLIT_ID}f1001_imaginary": Figure(),
            f"correction2{SPLIT_ID}f1010_imaginary": Figure(),
            f"correction2{SPLIT_ID}f1010_amplitude": Figure(),
            f"correction2{SPLIT_ID}f1010_real": Figure(),
            f"correction2{SPLIT_ID}f1010_phase": Figure(),
            "f1010_real_EXPDELTAREAL": Figure(),
            "f1010_imaginary_EXPDELTAIMAG": Figure(),
            "f1001_phase_EXPDELTAPHASE": Figure(),
            "f1001_amplitude_EXPDELTAAMP": Figure(),
        }

        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.TabWidget", MockTabAndWindow)
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.PlotWidget", MockPlotWidget)
        window = MockTabAndWindow("")
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.VerticalTabWindow", window)

        show_plots(figure_dict)

        assert window.title != ""  # set to something else
        assert window.shown

        assert len(window.tabs) == 3  # 1 for all, 2 for correction[12]
        for idx_tab, tab in enumerate(window.tabs):
            assert len(tab.tabs) == 2  # either 1 rdt 2 columns or 2 rdts 1 column
            if idx_tab:
                assert tab.title == f"correction{idx_tab}"

                for idx, plottab in enumerate(tab.tabs):
                    title_parts = plottab.title.split(" ")
                    rdt = "f1001" if idx_tab == 1 else "f1010"
                    assert len(title_parts) == 2
                    assert title_parts[0] == rdt
                    assert title_parts[1].lower() == ["a/p", "r/i"][idx]
            else:
                for idx, plottab in enumerate(tab.tabs):
                    assert plottab.title.lower() == ("f1010 r/i" if idx else "f1001 a/p")


    @pytest.mark.basic
    def test_pyplot(self, monkeypatch):
        figure_dict = {
            "param1_x_PARAM1X": Figure(),
            f"param1{SPLIT_ID}PARAM1Y": Figure(),
        }

        result_dict = {}
        def mock_fun(fig, title):
            result_dict[title] = fig

        fake_plt = MockTabAndWindow("")
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.VerticalTabWindow", None)
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.plt", fake_plt)
        monkeypatch.setattr("omc3.plotting.plot_checked_corrections.create_pyplot_window_from_fig", mock_fun)

        show_plots(figure_dict)

        assert fake_plt.shown
        for title in figure_dict:
            new_title = title.replace(SPLIT_ID, " ")
            assert result_dict[new_title] is figure_dict[title]


 # Helper ----------------------------------------------------------------------

@dataclass
class MockTabAndWindow:
    title: str
    tabs: list = field(default_factory=list, init=False)
    shown: bool = False

    def __call__(self, title: str):
        self.title = title
        return self

    def add_tab(self, tab: "MockTabAndWindow"):
        self.tabs.append(tab)

    def show(self):
        self.shown = True


@dataclass
class MockPlotWidget:
    fig1: Figure
    fig2: Figure = None
    title: str = ""


def _create_fake_measurement(tmp_path: Path, model_path: Path, twiss_path: Path):
    model_df = tfs.read(model_path / TWISS_DAT, index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_df = tfs.read(twiss_path, index=NAME)
    twiss_df = add_coupling_to_model(twiss_df)

    # create fake measurement data
    fake_measurement(
        model=model_df,
        twiss=twiss_df,
        randomize=[],
        outputdir=tmp_path,
    )

    # Add beta kmod files to the outputdir
    kmod_x_path = twiss_path.parent.parent / "beta_kmod_x.tfs"
    kmod_y_path = twiss_path.parent.parent / "beta_kmod_y.tfs"

    shutil.copy(kmod_x_path, tmp_path / kmod_x_path.name)
    shutil.copy(kmod_y_path, tmp_path / kmod_y_path.name)