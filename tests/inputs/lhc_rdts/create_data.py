# Convert a MAD-NG output into a fake measurement, then use OMC3 to calculate this fake measurement.
import time
import tfs

from optics_functions.rdt import calculate_rdts
from tests.inputs.lhc_rdts.MAD_helpers import (
    convert_tfs_to_madx,
    create_model_dir,
    get_twiss_elements,
    run_twiss_rdts,
    save_analytical_model,
    save_ng_model,
    save_x_model,
    to_ng_rdts,
    write_tbt_file,
)
from tests.inputs.lhc_rdts.omc3_helpers import (
    filter_IPs,
    get_rdts,
    run_harpy,
    get_rdts_from_harpy,
    get_file_suffix,
)
from tests.inputs.lhc_rdts.rdt_constants import DATA_DIR, MODEL_NG_PREFIX

run_madng = True
save_omc3_analysis = True
run_analytical_model = True

if run_madng:
    for beam in [1, 2]:
        for order in [2, 3]:
            # Create the model to this specific beam, order
            create_model_dir(beam, order)

            # Retrieve the RDTs for this specific beam and order
            ng_rdts = to_ng_rdts(get_rdts(beam, order))

            # Run MAD-NG twiss to get the RDTs
            model_ng = run_twiss_rdts(beam, ng_rdts, order) 
            
            # Convert the MAD-NG output to a MAD-X style TFS (waiting for TFS update)
            model_ng = convert_tfs_to_madx(model_ng)

            # Remove the BPMs around the IP
            model_ng = filter_IPs(model_ng)

            # Save the model
            save_ng_model(model_ng, beam, order)
            
            # Run MAD-NG track to produce a TBT file
            write_tbt_file(beam, order)

            # Run Harpy to get the RDTs
            run_harpy(beam, order)

if run_analytical_model:
    for beam in [1, 2]:
        for order in [2, 3]:
            create_model_dir(beam, order)
            ng_rdts = to_ng_rdts(get_rdts(beam, order))
            analytical_df = get_twiss_elements(beam, order)
            analytical_df = convert_tfs_to_madx(analytical_df)
            analytical_df = calculate_rdts(analytical_df, ng_rdts, feeddown=1)
            analytical_df = filter_IPs(analytical_df) 
            save_analytical_model(analytical_df, beam, order)

if save_omc3_analysis:
    for beam in [1, 2]:
        for order in [2, 3]:
            
            print("Running analysis")
            analysis_runtime = time.time()
            rdt_dfs = get_rdts_from_harpy(beam, order)
            print(f"Analysis Runtime: {time.time() - analysis_runtime}")
            
            file_ext = get_file_suffix(beam, order)
            model_ng = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_ext}.tfs", index="NAME")

            for rdt, rdt_df in rdt_dfs.items():
                assert len(model_ng.index.intersection(rdt_df.index)) == len(
                    rdt_df.index
                ), "Not all BPMs are in the model_ng analysis"

                # Now we know all the BPMs are the same, make sure they are in the same order
                rdt_dfs[rdt] = rdt_df.loc[model_ng.index]

            save_x_model(rdt_dfs, beam, order) 
            # Remove some unnecessary folders and files
        print("Done with order", order, "beam", beam)

print("Script finished")
