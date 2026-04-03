# Random Forest for Rasters
These functions facilitate applying the ```scikit-learn``` [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to spatial data. Random forest is a machine learning algorithm that uses decision trees to classify observations. It uses ensemble learning, an term which refers to algorithms that combine multiple models to produce their final model. In a random forest algorithm, each decision tree learns to make decisions about an observation's class based on its features (i.e., data associated with each observation). The final model then combines all of those decision trees when making predictions.

# Limitations
These functions are mostly focused on transforming spatial data into a format to which this classifier can be applied. However, when fitting the data to the classifier, these functions aren't inherently making any measure of observation spatial distribution. [Observations that are very close together will likely share similar characteristics](https://en.wikipedia.org/wiki/Tobler%27s_first_law_of_geography). Therefore, very close together observations can't necessarily be considered independent observations. 

# Other Implementations
I wrote these functions when I was trying to teach myself about random forest and am making them available in case other researchers may find them useful. However, there are other, full featured libaries/implementations for applying random forest to spatial data which may be better suited to your purposes. For instance, as discussed in the Limitations section is above, these functions do not inherently allow users to assess spatial autocorrelation errors or doing any spatial-specific tuning. Other implementations include options in:
* Python - [PyGRF](https://doi.org/10.1111/tgis.13248) (see [repository](https://github.com/geoai-lab/PyGRF)) a full-featured implementation developed by the University at Buffalo's GeoAI lab 
* ArcGIS Pro - see options in the [Spatial Statistics toolbox](https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-statistics/forestbasedclassificationregression.htm)
* R - [spatialRF](https://blasbenito.github.io/spatialRF/). This package documentation also has a great discussion of [spatial autocorrelation](https://blasbenito.github.io/spatialRF/articles/spatial_models.html) and how this may bias the model.
* [Google Earth Engine](https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest)

# Usage
I wrote these functions with presence/absence predictive modelling in mind--specifically, using archaeological site location data and landscape variables to predict the location of other, similar sites. In what follows, I refer to site locations as "presence" data and associated variables (e.g., slope, precipitation) as "features."

These functions work with raster data: all feature data should be in raster format, as should all presence data. Typically, presence data would be in a vector format (site location points or site boundary polygons). These can be converted in any GIS software (or using Python GIS libraries) to a raster where each cell has a defined value (1) if the cell contains a site.

The ```utils/spatial_functions.py``` script contains helper functions to transform these raster data into a ```pandas``` DataFrame, fit the data to the RandomForestClassifier, and then test the classifer on raster data outside of the training area and export the results as a georeferenced raster. The Usage Example section below walks through an example of applying these functions (on fake data).

## Usage Example

### Creating feature data and "presence" data
To test these functions, I created a dataset of fake site locations (that way I could share images of site distributions without concern about site sensitivity) on top of real environmental data. In theory, other users of this repository would have real data to which they could apply these functions. However, I still explain here in detail how I generated the fake data in case it might be useful for others. All initial data preparation was completed in QGIS (analagous functions exist for all these steps in ArcGIS Pro). Skip to **Read in Data** if your data is ready to go.

#### Features
I selected a study area, downloaded 10 m DEMs covering that study area, and merged them into a single raster (data available from the [USGS National Map Data Downloader](https://apps.nationalmap.gov/downloader/); look for 1/3 arc second data under the 3D Elevation Program header). Using the raster terrain analysis tools in QGIS, I then generated a slope raster, aspect raster, and terrain ruggedness index raster with this merged DEM as the input. Using the [r.stream.extract](https://grass.osgeo.org/grass78/manuals/r.stream.extract.html) tool in the GRASS toolbox, I generated a raster of all streams with an initiation threshold of 1000 pixels, reclassified the raster so each unique stream was represented by pixels with a value of 1, and used the Proximity (Raster Analysis) tool in the GDAL Raster Analysis toolbox to generate a raster representing (Euclidean) distance to each of these stream pixels. I also downloaded the 800 m resolution [PRISM climate normals](https://prism.oregonstate.edu/normals/) precipitation data and clipped it to the same study area. These rasters all represent landscape features that could, in theory, have influenced the placement of an archaeological site. 

#### "Presence" data
I then generated 1500 random points within the extent of the merged DEMs (Random Points in Extent tool in QGIS). I then used the Sample Raster Values tool in the QGIS Raster Analysis toolbox to determine the slope, aspect, TRI, stream proximity, and precipitation value for each point. Then I applied *k*-means clustering, an unsupervised classification algorithm, with *k* = 3 to identify groups within these data that shared characteristics in multivariate space. Cluster 1 became my "outlying site" locations and I selected 30 random points within Cluster 2 to become my "urban center" locations. Using the GRASS tool r.cost, I used the slope raster as a cost surface and these 30 points as "start points" to generate a raster representing cost of travel to these urban centers across the entire study area. This became the final feature ("cost distance to large sites").

#### "Absence" data
The random forest classifier will use the site location data (presence data) and associated variables to build decision trees that predict site locations. However, it's also useful to provide the model with absence data - locations that do not contain the thing you're looking for. The classifier will only be seeing information about the landscape that we're giving it, and so a model trained on presence-only data could be less precise. For this example, I simply picked the random points from Cluster 3 to become my absence points. These should be more similar to each other than to the points in the presence layer such that they are useful for the model. In a real-world scenario, you are more likely to have reliable presence data than reliable absence data. Biases and gaps in survey coverage, for instance, mean we can be more confident about where sites *are* in a given study area, but less sure about where they are not. For "real" absence data, you could generate a random sample of points, ensuring they do not overlap with real site points, or place points manually based on areas where prior surveys suggest there are indeed no sites.

#### Final analysis raster(s)
All the functions in ```utils/spatial_functions.py``` work with raster data. Prior to analysis, I converted the presence and absence points into separate rasters using the Rasterize tool in QGIS, using "1" as a value to burn and selecting the same extent and resolution as the slope, aspect etc. rasters. I also ensure that all rasters covered the same extent and were aligned (this necessitated resampling the PRISM precipitation raster to a 10 m resolution prior to clipping and aligning). For this example, I then used the Merge tool in the QGIS Raster Analysis toolbox with the presence raster, absence raster, and all six feature rasters as inputs. I selected the "Place each input file into a separate band" option to make a single, eight-band analysis raster.

### Read in data
If working with a multi-band analysis raster, use the ```get_bands()``` function to read each band as its own raster. This function returns a list of ```numpy``` arrays (i.e., each raster band opened and read in with ```rasterio```):

```
analysis_raster = "test-data/Analysis_layers_merged.tif"
number_of_bands = 8
presence, absence, aspect, cost_large_site, prcp, slope, stream_dist, tri = get_bands(analysis_raster, number_of_bands)
```

I knew the order of the bands within the raster and used that to assign descriptive variable names. 

Alternatively, if you don't have your data stacked into a single raster, you could simply open and read each raster individually (or in a for loop) with ```rasterio```:

```
presence = rasterio.open('test-data/Presence.tif').read(1)
absence = rasterio.open('test-data/Presence.tif').read(1)
#etc.
```

### Build DataFrame from rasters
The function ```build_pa_df()``` uses the presence and absence rasters to get values associated with each feature for each presence or absence cell and turns that into a single DataFrame. Each row in the DataFrame is a single observation.

```
#list of opened rasters, one for each feature
feature_list = [aspect, cost_large_site, prcp, slope, stream_dist, tri]
#list of strings representing the desired output column names, in the same order as feature_list
feature_names = ['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI']
final_df = build_pa_df(feature_list, presence, absence, feature_columns)
```

### Train model and view performance metrics
The function ```train_predict_results()``` instantiates the RandomForestClassifier, divides the data in the analysis DataFrame into training and testing datasets, fits the classifier to the training data, and prints model performance metrics (precision, recall, F1) on the testing dataset by default:

```
rforest, xtest, ytest = train_predict_results(final, ['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI'])
```

This function returns the trained classifier (```rforest```) and the testing dataset for additional metric calculations or for examining feature importance.

### Predict on new data
After training, this trained classifier can be applied to new data. The function ```predict_on_mb_raster()``` takes the new multiband raster and applies the classifier to each cell in a vectorized manner, returning a single array with likely presence represented as "1" and absence as "0."

```
#rforest is the trained classifier - the output of train_predict_results()
#No need to read in bands separately in this case; all of that is done within the function
prediction_raster = 'test-data/Area_to_analyze.tif'
prediction = predict_on_mb_raster(prediction_raster, num_bands = 6, classifier = rforest)
```

The function ```predict_on_rasters()``` works with single-band rasters. In this case, as before, read in all the rasters individually, then provide a list of those arrays to the function.

```
prediction = predict_on_rasters(read_raster_list_ordered = [aspect, cost_large_site, prcp, slope, stream_dist, tri], classifier = rforest)
```

### Save output
The function ```save_raster_prediction()``` uses rasterio to save the output prediction array as a GeoTIFF with the same extent and resolution as the input:

```
prediction_raster = 'test-data/Area_to_analyze.tif'
prediction_out = 'test-data/Prediction.tif'
save_raster_prediction(prediction_raster, prediction_out)
```

These steps are wrapped in a function for ease of use, but this part is relatively straightforward and follows the typical procedure for exporting a raster from an array (see [rasterio documentation](https://rasterio.readthedocs.io/en/stable/topics/writing.html)).

### Optional: Examining feature importance
There are a couple ways to examine how the random forest classifier is making its predictions. We can look at individual decision trees using:

```
import matplotlib.pyplot as plt
sklearntree.plot_tree(rforest.estimators_[1],
                  feature_names = ['Aspect', 'Lg_site_dist', 'PRCP', 'Stream_dist', 'Slope', 'TRI'],
                  filled = True)
plt.show()
```

But we can also look at the relative importance of different features in the final model. I do not have dedicated functions for this in the repository, but this process is relatively straightforward following the ```scikit-learn``` [documentation](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html). The trained classifier contains feature importance scores based on a calculation of mean decrease in impurity. These can be accessed from the classifier at ```rforest.feature_importances_```. However, these MDI scores are subject to bias, specifically towards high cardinality features (i.e., features with more distinct elements). An alternate way to calculate feature importance is to use the test dataset and calculate permutation importance on the test dataset. This procedure takes the features, calculates the classifier's score (e.g., accuracy) when predicting with those features, then it permutes (i.e., randomly shuffles) a column and sees how much the performance changes. 

```
perm = permutation_importance(rforest, xtest, ytest) #By default, this runs five permutations for each feature
```
Features that change the score more than others are more important to the decisions the classifier is making.
