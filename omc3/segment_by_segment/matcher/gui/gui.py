import sys
import os
import logging
import constants
from PyQt5 import QtWidgets
from omc3.segment_by_segment.matcher import log_handler
from omc3.segment_by_segment.matcher.gui.sbs_gui_main import SbSGuiMainController


LOGGER = logging.getLogger(__name__)


def main(lhc_mode=None, match_path=None, input_dir=None):
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    main_controller = SbSGuiMainController()
    if match_path is None or lhc_mode is None:
        lhc_mode, match_path = main_controller.ask_for_initial_config(
            lhc_mode,
            match_path,
        )
        if match_path is None or lhc_mode is None:
            return
    log_handler.add_file_handler(match_path)
    if lhc_mode not in constants.LHC_MODES:
        raise ValueError("Invalid lhc mode, must be one of " +
                         str(constants.LHC_MODES))
    LOGGER.info("-------------------- ")
    LOGGER.info("Configuration:")
    LOGGER.info("- LHC mode: " + lhc_mode)
    LOGGER.info("- Match output path: " + os.path.abspath(match_path))
    LOGGER.info("-------------------- ")
    main_controller.set_match_path(match_path)
    main_controller.set_lhc_mode(lhc_mode)
    main_controller.set_input_dir(input_dir)
    main_controller.show_view()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
