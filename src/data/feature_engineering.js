// ============================================================
// FractalProspectNet v4 — Tiled feature_stack export to GEE Assets (v2, fixed tile count)
// ============================================================
// ---- 1. Zimbabwe boundary (actual country geometry, not bounding box) ----
var zimGeom = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Zimbabwe'))
  .geometry();
// ---- 2. Load existing feature_stack, force Float32 everywhere ----
var featureStack = ee.Image('projects/mineral-prospectivity-zim/assets/feature_stack')
  .toFloat();
// ---- 3. Clip to actual country geometry ----
var featureStackClipped = featureStack.clip(zimGeom);
print('Band names:', featureStackClipped.bandNames());
print('Band count:', featureStackClipped.bandNames().size());
// ---- 4. Bounding box corners ----
var bbox = zimGeom.bounds();
var bboxCoords = ee.List(bbox.coordinates().get(0));
var minLon = ee.Number(ee.List(bboxCoords.get(0)).get(0)).getInfo();
var minLat = ee.Number(ee.List(bboxCoords.get(0)).get(1)).getInfo();
var maxLon = ee.Number(ee.List(bboxCoords.get(2)).get(0)).getInfo();
var maxLat = ee.Number(ee.List(bboxCoords.get(2)).get(1)).getInfo();
print('Bounding box:', minLon, minLat, maxLon, maxLat);
// ---- 5. Build coarser 1.0 degree grid, KEEP ONLY tiles with real overlap ----
var tileDeg = 1.0; // ~110km x 110km tiles — comfortably under EEException limits
var minOverlapKm2 = 50; // drop slivers / near-empty edge tiles
var lonStepsNum = Math.ceil((maxLon - minLon) / tileDeg);
var latStepsNum = Math.ceil((maxLat - minLat) / tileDeg);
print('Candidate grid dimensions (lon x lat):', lonStepsNum + ' x ' + latStepsNum);
print('Total candidate tiles before filtering:', lonStepsNum * latStepsNum);
var tileTasks = [];
var tileIndex = 0;
for (var i = 0; i < lonStepsNum; i++) {
  for (var j = 0; j < latStepsNum; j++) {
    var tileMinLon = minLon + (i * tileDeg);
    var tileMinLat = minLat + (j * tileDeg);
    var tileMaxLon = tileMinLon + tileDeg;
    var tileMaxLat = tileMinLat + tileDeg;
    var tileRect = ee.Geometry.Rectangle(
      [tileMinLon, tileMinLat, tileMaxLon, tileMaxLat],
      'EPSG:4326',
      false
    );
    var tileClipped = tileRect.intersection(zimGeom, ee.ErrorMargin(1));
    var overlapAreaKm2 = tileClipped.area(1).divide(1e6).getInfo();
    if (overlapAreaKm2 > minOverlapKm2) {
      tileTasks.push({
        index: tileIndex,
        geometry: tileClipped,
        areaKm2: overlapAreaKm2
      });
      tileIndex++;
    }
  }
}
print('Total tiles with real overlap (final count):', tileTasks.length);
print('Tile area summary (km2 per tile):', tileTasks.map(function(t) { return Math.round(t.areaKm2); }));
// ---- 6. Submit export tasks for each real tile ----
tileTasks.forEach(function(tile) {
  var tileImage = featureStackClipped.clip(tile.geometry);
  Export.image.toAsset({
    image: tileImage,
    description: 'feature_stack_tile_' + tile.index,
    assetId: 'projects/mineral-prospectivity-zim/assets/tiles/feature_stack_tile_' + tile.index,
    region: tile.geometry,
    scale: 30,
    crs: 'EPSG:4326',
    maxPixels: 1e10
  });
});
print('Export tasks created — go to the Tasks tab to run them.');
print('Final total tasks:', tileTasks.length);
// ---- 7. Optional later step: mosaic all tile assets for reference ----
// var tileAssetIds = tileTasks.map(function(t) {
//   return 'projects/mineral-prospectivity-zim/assets/tiles/feature_stack_tile_' + t.index;
// });
// var tileCollection = ee.ImageCollection(tileAssetIds);
// var mosaicked = tileCollection.mosaic();
