# Convert a MAD-NG output into a fake measurement, then use OMC3 to calculate this fake measurement.
import time

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
)

run_madng = False
save_omc3_analysis = False
run_analytical_model = True

if run_madng:
    for beam in [1, 2]:
        # Create the MAD-X model
        for is_skew in [False, True]:
            for order in [2, 3]:
                # Create the model to this specific beam, order and skew
                create_model_dir(beam, order, is_skew)

                # Retrieve the RDTs for this specific beam, order and skew
                ng_rdts = to_ng_rdts(get_rdts(order, is_skew))

                # Run MAD-NG twiss to get the RDTs
                model_ng = run_twiss_rdts(beam, ng_rdts, order, is_skew) 
                
                # Convert the MAD-NG output to a MAD-X style TFS (waiting for TFS update)
                model_ng = convert_tfs_to_madx(model_ng, beam)

                # Remove the BPMs around the IP
                model_ng = filter_IPs(model_ng)

                # Save the model
                save_ng_model(model_ng, beam, order, is_skew)
                
                # Run MAD-NG track to produce a TBT file
                write_tbt_file(beam, order, is_skew)

                # Run Harpy to get the RDTs
                run_harpy(beam, order, is_skew)

if run_analytical_model:
    for beam in [1, 2]:
        for is_skew in [False, True]:
            for order in [2, 3]:
                create_model_dir(beam, order, is_skew)
                ng_rdts = to_ng_rdts(get_rdts(order, is_skew))
                analytical_df = get_twiss_elements(beam, order, is_skew)
                analytical_df = convert_tfs_to_madx(analytical_df, beam)
                analytical_df = calculate_rdts(analytical_df, ng_rdts)
                analytical_df = filter_IPs(analytical_df) 
                save_analytical_model(analytical_df, beam, order, is_skew)

if save_omc3_analysis:
    for beam in [1, 2]:
        for is_skew in [False, True]:
            for order in [2, 3]:
                analysis_runtime = time.time()
                print("Running analysis")
                rdt_dfs = get_rdts_from_harpy(beam, order, is_skew)
                print(f"Analysis Runtime: {time.time() - analysis_runtime}")

                for rdt, rdt_df in rdt_dfs.items():
                    assert len(model_ng.index.intersection(rdt_df.index)) == len(
                        rdt_df.index
                    ), "Not all BPMs are in the model_ng analysis"

                    # Now we know all the BPMs are the same, make sure they are in the same order
                    rdt_dfs[rdt] = rdt_df.loc[model_ng.index]

                save_x_model(rdt_dfs, beam, order, is_skew) 
                # Remove some unnecessary folders and files
            print("Done with order", order, "is_skew", is_skew, "beam", beam)

print("Script finished")
