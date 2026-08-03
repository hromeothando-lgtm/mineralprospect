\# Terrain Branch — GEE Scripts (DISCONTINUED BRANCH)



\*\*Status: permanently discontinued.\*\* This branch was dropped from the FractalProspect-Net v4

architecture after pysheds, WhiteboxTools, richdem, and numba all failed to produce a working

country-scale pipeline (Jacobi-style pit-filling was non-convergent; WhiteboxTools has an

unresolved GeoTIFF codec bug; heap-overflow issues on the full-extent DEM). The current

architecture is locked at two branches (Spectral + Stochastic). This documentation is

preserved for the Methods/limitations scope-narrowing narrative, not as active pipeline

documentation — do not resume this branch without new explicit cause.



\## dem\_download\_buffered.js

Exports Copernicus GLO-30 DEM buffered 2km beyond Zimbabwe's true national boundary,

solving the flow-accumulation edge-effect problem (pysheds defaults nodata to 0,

causing false drainage routing at the country border).



Outputs:

\- GEE Asset: projects/mineral-prospectivity-zim/assets/zimbabwe\_dem\_glo30\_buffered2km

\- Drive backup: mineral\_prospectivity\_zim/zimbabwe\_dem\_glo30\_buffered2km\_drive.tif

\- GEE Asset: projects/mineral-prospectivity-zim/assets/zimbabwe\_true\_boundary

&#x20; (used to clip final terrain bands back to true Zimbabwe extent after pysheds processing)



Run in GEE Code Editor, confirm both exports in the Tasks tab before running

country-scale pysheds pipeline in Colab.

