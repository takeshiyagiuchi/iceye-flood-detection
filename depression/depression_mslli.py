from whitebox_workflows import WbEnvironment


def depression_mslli(
        filepath_input_dem: str,
        filepathbase_out_depression_mslli: str,
        filepathbase_out_depression_mslliks: str,
        min_scale: int,
        step_size: int,
        num_steps: int,
        step_nonlinearity: float
) -> None:
    """
    Compute the Multiscale Low-Lying Index (MSLLI) from a DEM/DTM.

    This function uses WhiteboxTools to calculate:
    - index: multiscale low-lying index, representing how relatively low each pixel
             is compared to its surroundings across multiple spatial scales
    - key_scale: the scale at which the strongest low-lying response occurs

    The MSLLI provides a continuous measure of "depression likelihood", where
    higher values typically indicate areas that are more likely to accumulate water
    (e.g., flood-prone or low-lying regions).

    Parameters
    ----------
    filepath_input_dem : str
        Path to the input Digital Elevation Model (DEM) or Digital Terrain Model (DTM) raster.
    filepathbase_out_depression_mslli : str
        Output pathbase for the multiscale low-lying index raster.
    filepathbase_out_depression_mslliks : str
        Output pathbase for the key scale raster (scale of strongest response).
    min_scale : int
        Minimum filter/window size (in pixels) for multiscale analysis.
    step_size : int
        Increment between successive scales.
    num_steps : int
        Number of scale steps to evaluate.
    step_nonlinearity : float
        Controls how scales are distributed:
        - 1.0 → linear spacing
        - >1.0 → emphasizes larger scales
        - <1.0 → emphasizes smaller scales

    Returns
    -------
    None
        Results are written to disk as raster files.
    """
    try:
        # Build out file paths
        filepath_out_depression_mslli = filepathbase_out_depression_mslli + \
            f"_min{min_scale}" + \
            f"_max{min_scale + step_size * num_steps}" + \
            f"_stp{step_size}" + \
            f"_lin{int(step_nonlinearity * 100):03d}" + \
            ".tif"
        print(f"filepath_out_depression_mslli: {filepath_out_depression_mslli}")
        filepath_out_depression_mslliks = filepathbase_out_depression_mslliks + \
            f"_min{min_scale}" + \
            f"_max{min_scale + step_size * num_steps}" + \
            f"_stp{step_size}" + \
            f"_lin{int(step_nonlinearity * 100):03d}" + \
            ".tif"
        print(f"filepath_out_depression_mslliks: {filepath_out_depression_mslliks}")

        # Initialize Whitebox environment
        wbe = WbEnvironment()
        wbe.working_directory = "."  # Set working directory
        wbe.verbose = True           # Enable progress/output logging

        # Read input DEM/DTM raster
        dem = wbe.read_raster(file_name=filepath_input_dem)

        # Compute multiscale low-lying index
        # index: degree to which a pixel is lower than its surroundings across scales
        # key_scale: scale at which the maximum low-lying response occurs
        index, key_scale = wbe.multiscale_low_lying_index(
            dem=dem,
            min_scale=min_scale,
            step_size=step_size,
            num_steps=num_steps,
            step_nonlinearity=step_nonlinearity
        )

        # Write output rasters to disk
        wbe.write_raster(index, filepath_out_depression_mslli, True)
        wbe.write_raster(key_scale, filepath_out_depression_mslliks, True)

    except Exception as e:
        print("Error:", e)

    finally:
        print("Finished")


if __name__ == "__main__":
    # Example execution with specified input/output paths and scale parameters
    depression_mslli(
        filepath_input_dem="data/merged_dtm_elora.tif",
        filepathbase_out_depression_mslli="data/depression_mslli",
        filepathbase_out_depression_mslliks="data/depression_mslliks",
        min_scale=5,
        step_size=5,
        num_steps=900,
        step_nonlinearity=1.0,
    )
