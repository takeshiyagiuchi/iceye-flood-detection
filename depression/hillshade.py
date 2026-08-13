from whitebox_workflows import WbEnvironment


def hillshade(
        filepath_input_dem: str,
        filepathbase_out_hillshade: str,
        azimuth: float = 315.0,
        altitude: float = 30.0,
        z_factor: float = 1.0,
) -> None:
    """
    Generate a hillshade raster from an input DEM/DTM using WhiteboxTools.

    Hillshade is a shaded relief representation of terrain that simulates
    illumination from a light source defined by azimuth and altitude angles.
    It is commonly used for terrain visualization and qualitative analysis
    of topographic features.

    Parameters
    ----------
    filepath_input_dem : str
        Path to the input DEM/DTM raster file.
    filepathbase_out_hillshade : str
        Base path (without extension) for the output hillshade raster.
        Parameter values will be appended to the filename.
    azimuth : float, optional
        Illumination direction in degrees clockwise from north (default: 315.0).
    altitude : float, optional
        Illumination angle above the horizon in degrees (default: 30.0).
    z_factor : float, optional
        Vertical exaggeration factor applied to elevation values (default: 1.0).

    Returns
    -------
    None
        The output raster is written to disk.
    """
    try:
        # Build output file path with parameter encoding for reproducibility
        filepath_out_hillshade = filepathbase_out_hillshade + \
            f"_azm{int(azimuth)}" + \
            f"_alt{int(altitude)}" + \
            f"_z{int(z_factor * 100):03d}" + \
            ".tif"
        print(f"filepath_out_hillshade: {filepath_out_hillshade}")

        # Initialize Whitebox environment
        wbe = WbEnvironment()
        wbe.working_directory = "."  # Set working directory
        wbe.verbose = True           # Enable progress and diagnostic logging

        # Read input DEM/DTM raster
        dem = wbe.read_raster(file_name=filepath_input_dem)

        # Compute hillshade.
        # Hillshade simulates illumination based on terrain slope and aspect,
        # producing a grayscale image that enhances terrain structure.
        hillshade_out = wbe.hillshade(
            dem=dem,
            azimuth=azimuth,
            altitude=altitude,
            z_factor=z_factor
        )

        # Write output raster to disk
        wbe.write_raster(hillshade_out, filepath_out_hillshade, True)

    except Exception as e:
        print("Error:", e)

    finally:
        print("Finished")


if __name__ == "__main__":
    # Example execution with specified input/output paths and parameters
    hillshade(
        filepath_input_dem="data/merged_dtm_elora.tif",
        filepathbase_out_hillshade="data/hillshade",
        azimuth=315.0,
        altitude=30.0,
        z_factor=1.0,
    )
