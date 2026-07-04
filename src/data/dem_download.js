var zimbabwe = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Zimbabwe'));

var zimGeom = zimbabwe.geometry();

var bufferDistanceMeters = 2000;

var bufferedGeom = zimGeom.buffer(bufferDistanceMeters);

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .select('DEM')
  .mosaic()
  .setDefaultProjection('EPSG:4326', null, 30)
  .toFloat();

var demBuffered = dem.clip(bufferedGeom);

Map.centerObject(zimbabwe, 6);
Map.addLayer(bufferedGeom, {color: 'red'}, 'Buffered boundary (2km)', false);
Map.addLayer(zimGeom, {color: 'blue'}, 'True Zimbabwe boundary', false);
Map.addLayer(demBuffered, {min: 200, max: 2500, palette: ['blue', 'green', 'yellow', 'brown', 'white']}, 'Buffered DEM');

Export.image.toAsset({
  image: demBuffered,
  description: 'zimbabwe_dem_glo30_buffered2km',
  assetId: 'projects/mineral-prospectivity-zim/assets/zimbabwe_dem_glo30_buffered2km',
  region: bufferedGeom,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: demBuffered,
  description: 'zimbabwe_dem_glo30_buffered2km_drive',
  folder: 'mineral_prospectivity_zim',
  region: bufferedGeom,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.table.toAsset({
  collection: ee.FeatureCollection([ee.Feature(zimGeom)]),
  description: 'zimbabwe_true_boundary',
  assetId: 'projects/mineral-prospectivity-zim/assets/zimbabwe_true_boundary'
});

print('Buffered geometry area (sq km):', bufferedGeom.area().divide(1e6));
print('True Zimbabwe area (sq km):', zimGeom.area().divide(1e6));
