from whitebox_workflows import WbEnvironment


def depression_devmax(
        filepath_input_dem: str,
        filepath_out_depression_devmax_base: str,
        filepath_out_depression_devscale_base: str,
        min_scale: int,
        max_scale: int,
        step_size: int
) -> None:
    """
    Compute multiscale elevation deviation (MaxElevationDeviation) from a DEM/DTM.

    This function uses WhiteboxTools to calculate:
    - dev_max: maximum elevation deviation (indicates relative depressions/elevations)
    - dev_scale: scale at which the maximum deviation occurs

    These outputs can be used to identify low-lying areas (potential depressions)
    across multiple spatial scales.

    Parameters
    ----------
    filepath_input_dem : str
        Path to the input Digital Elevation Model (DEM) or Digital Terrain Model (DTM) raster.
    filepath_out_depression_devmax_base : str
        Output pathbase for the maximum elevation deviation raster.
    filepath_out_depression_devscale_base : str
        Output pathbase for the scale raster corresponding to max deviation.
    min_scale : int
        Minimum filter/window size (in pixels) for multiscale analysis.
    max_scale : int
        Maximum filter/window size (in pixels) for multiscale analysis.
    step_size : int
        Step size between scales.

    Returns
    -------
    None
        Results are written to disk as raster files.
    """
    try:
        # Build out file paths
        filepath_out_depression_devmax = filepath_out_depression_devmax_base + \
            f"_min{min_scale}" + \
            f"_max{max_scale}" + \
            f"_stp{step_size}" + \
            ".tif"
        print(f"filepath_out_depression_devmax: {filepath_out_depression_devmax}")
        filepath_out_depression_devscale = filepath_out_depression_devscale_base + \
            f"_min{min_scale}" + \
            f"_max{max_scale}" + \
            f"_stp{step_size}" + \
            ".tif"
        print(f"filepath_out_depression_devscale: {filepath_out_depression_devscale}")

        # Initialize Whitebox environment
        wbe = WbEnvironment()
        wbe.working_directory = "."  # Set working directory
        wbe.verbose = True           # Enable progress/output logging

        # Read input DTM raster
        dtm = wbe.read_raster(file_name=filepath_input_dem)

        # Compute multiscale elevation deviation
        # dev_max: magnitude of deviation (negative = low-lying areas)
        # dev_scale: scale at which the maximum deviation occurs
        dev_max, dev_scale = wbe.max_elevation_deviation(
            dem=dtm,
            min_scale=min_scale,
            max_scale=max_scale,
            step_size=step_size
        )

        # Write output rasters to disk
        wbe.write_raster(dev_max, filepath_out_depression_devmax, True)
        wbe.write_raster(dev_scale, filepath_out_depression_devscale, True)

    except Exception as e:
        print("Error:", e)

    finally:
        print("Finished")


if __name__ == "__main__":
    # Example execution with specified input/output paths and scale parameters
    depression_devmax(
        filepath_input_dem="data/merged_dtm_elora.tif",
        filepath_out_depression_devmax_base="data/depression_devmax",
        filepath_out_depression_devscale_base="data/depression_devscale",
        min_scale=1,
        max_scale=100,
        step_size=1
    )
