"""
Creating data for lhc_rdts test
-------------------------------

This script creates the data required for the lhc_rdts test. The order of operations is as follows:
1. Create the model for the specific beam, these are stored in seperate folders.
2. Retrieve the RDTs - could be a constant. Is a function for flexibility.
3. Run MAD-NG twiss to get the values of the RDTs.
4. Convert the MAD-NG output to a MAD-X style TFS. This must be done as OMC3 only
reads MAD-X TFS files for the model, and expects some things about the file.
Also, this function reduces the file size.
5. Remove the BPMs around the IP, as due to the phase advances, these RDTs are
less accurate when calculated from OMC3.
6. Save just the RDTS to a TFS file.
7. Run MAD-NG track to produce a TBT file - a fake measurement.
8. Run the analytical model to get the RDTs.
9. Other OMC3 analysis - this is only required if you want to analyse everything rather than just run the test.
"""
import time

import tfs
from optics_functions.rdt import calculate_rdts

from tests.inputs.lhc_rdts.helper_lhc_rdts import (
    convert_tfs_to_madx,
    create_model_dir,
    get_model_dir,
    get_twiss_elements,
    run_twiss_rdts,
    save_analytical_model,
    save_ng_model,
    save_x_model,
    to_ng_rdts,
    write_tbt_file,
)
from tests.utils.compression import compress_model
from tests.utils.lhc_rdts.constants import DATA_DIR, MODEL_NG_PREFIX
from tests.utils.lhc_rdts.functions import (
    filter_out_BPM_near_IPs,
    get_file_suffix,
    get_rdt_names,
    get_rdts_from_optics_analysis,
    run_harpy,
)

run_madng = True
run_analytical_model = True
save_omc3_analysis = False # Only required if you want to analyse everything rather than just run the test.

for beam in [1, 2]:
    # Note: order is the order of the magnetic strength, not the order of the RDTs
    # For example, order 2 means k2(s) and order 3 means k3(s) with leads to RDTs of order 3 and 4
    # Create the model to this specific beam, order
    create_model_dir(beam)

    # Compress the model for source control
    compress_model(get_model_dir(beam))

    # Retrieve the RDTs for this specific beam and order and convert them into the MAD-NG format
    ng_rdts = to_ng_rdts(get_rdt_names())

    # Run MAD-NG twiss to get the RDTs
    model_ng = run_twiss_rdts(beam, ng_rdts)

    # Convert the MAD-NG output to a MAD-X style TFS (waiting for TFS update)
    model_ng = convert_tfs_to_madx(model_ng)

    # Remove the BPMs around the IP
    model_ng = filter_out_BPM_near_IPs(model_ng)

    # Save the model
    save_ng_model(model_ng, beam)

    if run_madng:
        # Run MAD-NG track to produce a TBT file
        write_tbt_file(beam)

    if run_analytical_model:
        analytical_df = get_twiss_elements(beam)
        analytical_df = convert_tfs_to_madx(analytical_df)
        analytical_df = calculate_rdts(analytical_df, ng_rdts, feeddown=2)
        analytical_df = filter_out_BPM_near_IPs(analytical_df)
        save_analytical_model(analytical_df, beam)

    if save_omc3_analysis:
        print("Running Harpy")
        # Run Harpy to get the RDTs
        run_harpy(beam)

        print("Running analysis")
        analysis_runtime = time.time()
        rdt_dfs = get_rdts_from_optics_analysis(beam)
        print(f"Analysis Runtime: {time.time() - analysis_runtime}")

        file_ext = get_file_suffix(beam)
        model_ng = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_ext}.tfs", index="NAME")

        for rdt, rdt_df in rdt_dfs.items():
            assert len(model_ng.index.intersection(rdt_df.index)) == len(
                rdt_df.index
            ), "Not all BPMs are in the model_ng analysis"

            # Now we know all the BPMs are the same, make sure they are in the same order
            rdt_dfs[rdt] = rdt_df.loc[model_ng.index]

        save_x_model(rdt_dfs, beam)
    print("Done with beam", beam)

print("Script finished")
