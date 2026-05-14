from whitebox_workflows import WbEnvironment

wbe = WbEnvironment()

dtm = wbe.read_raster("data/merged_dtm_elora.tif")

out = wbe.max_elevation_deviation(
    dem=dtm,
    min_scale=1,
    max_scale=100,
)

print(out)

# wbe.write_raster(out, "data/depression_devmax.tiff", True)
