# Terrain Branch — GEE Scripts

## dem_download_buffered.js
Exports Copernicus GLO-30 DEM buffered 2km beyond Zimbabwe's true national boundary,
solving the flow-accumulation edge-effect problem (pysheds defaults nodata to 0,
causing false drainage routing at the country border).

Outputs:
- GEE Asset: projects/mineral-prospectivity-zim/assets/zimbabwe_dem_glo30_buffered2km
- Drive backup: mineral_prospectivity_zim/zimbabwe_dem_glo30_buffered2km_drive.tif
- GEE Asset: projects/mineral-prospectivity-zim/assets/zimbabwe_true_boundary
  (used to clip final terrain bands back to true Zimbabwe extent after pysheds processing)

Run in GEE Code Editor, confirm both exports in the Tasks tab before running
country-scale pysheds pipeline in Colab.
