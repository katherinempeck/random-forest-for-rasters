from utils.spatial_functions import *

from rasterio.plot import reshape_as_image

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#This example is also summarized in the README

##################
## Read in Data ##
##################

#Set to multiband raster file path (with training data)
analysis_raster = "test-data/Analysis_layers_merged.tif"

#Alternatively, if you have separate rasters, you can open and read them individually with rasterio and pass them as a list to predict_on_rasters()
#This means that you could also do operations on elevation rasters within Python

#In this multiband raster, the first two bands are the presence/absence bands, the remainder are the data features
presence, absence, aspect, cost_large_site, prcp, slope, stream_dist, tri = get_bands(analysis_raster, 8)

#Visualize
fig, ax = plt.subplots(2, 3, figsize = (10, 10))
for r, t, a in zip([slope, aspect, tri, prcp, stream_dist, cost_large_site], 
                   ['Slope', 'Aspect', 'TRI', 'Annual Prcp. Normals (mm)', 'Stream Proximity', 'Large Site Travel Cost'],
                   ax.reshape(-1)):
    pos = a.imshow(r, cmap = 'viridis')
    fig.colorbar(pos, ax=a)
    a.axis('off')
    a.set_title(t)
fig.tight_layout()
plt.savefig('figs/Figure_1.jpg', bbox_inches = 'tight')

##################################
## Build DataFrame from Rasters ##
##################################

#Feature list constructed from above variables
feature_list = [aspect, cost_large_site, prcp, slope, stream_dist, tri]
#Names of columns, in the same order as the feature list
feature_columns = ['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI']

#Build analysis DF
final = build_pa_df(feature_list, presence, absence, feature_columns)

## Train Model and View Performance Metrics ##
rforest, xtest, ytest = train_predict_results(final, ['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI'])

#########################
## Predict on New Data ##
#########################

#Define prediction raster
prediction_raster = "test-data/Prediction_raster_merged.tif"
#This is a 6-band raster with bands in the same order as the main analysis raster (aspect, travel cost, prcp, slope, stream distance, tri)

#Set file path to save georeferenced prediction raster
prediction_out = 'test-data/Prediction.tif'

#Run prediction (using the classifier trained above)
#This works by using np.apply_along_axis() to apply the prediction function in a vectorized manner across all bands of the raster
    #There may be other ways to approach this 
#This gives us a binary array, with 1 representing predicted presence and 0 representing predicted absence

start = time.time()
prediction = predict_on_mb_raster(prediction_raster, num_bands = 6, classifier = rforest)
end = time.time()

fig, ax = plt.subplots()
image = ax.imshow(reshape_as_image(prediction), alpha = 0.5, cmap = 'viridis')
#Categorical legend trick from https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
values = np.unique(prediction.ravel())
colors = [image.cmap(image.norm(value)) for value in values]
patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i]) ) for i in range(len(values)) ]
ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad = 0)
ax.axis('off')
ax.set_title(f'Prediction\nElapsed: {round(end - start, 2)} seconds')
plt.savefig('figs/Figure_2.jpg', bbox_inches = 'tight')

###############
## Save Data ##
###############

#Write output to file
save_raster_prediction(prediction_raster, prediction_out, prediction)

###################################
## Optional - Feature Importance ##
###################################

#See https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html

#Built in .feature_importances_ is based on mean decrease in impurity
feature_importances = (rforest.feature_importances_*100).tolist()
features = ['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI']

mdi_df = pd.DataFrame([[f, i] for f, i in zip(features, feature_importances)], columns = ['Feature', 'Importance'])
mdi_df = mdi_df.sort_values(by = 'Importance', ascending = True)

print(f'MDI:\n{mdi_df}\n')

#As discussed in the scikit-learn documentation, this way of calculating feature importance is subject to bias, specifically towards high cardinality features
    #i.e., features with more distinct elements
#In their example, a feature composed entirely of random numbers was ranked as the third most important variable

#An alternate way to calculate feature importance is to use the test dataset and calculate permutation importance
#This takes the features, calculates the classifier's score (e.g., accuracy) on predicting with those features
#Then it permutes (i.e., randomly shuffles) a column and sees how much the performance changes
#By default, this happens five times (that's why there are five rows in the output dataframe)

n_repeats = 10
perm = permutation_importance(rforest, xtest, ytest, n_repeats = n_repeats)
sorted_importances_idx = perm.importances_mean.argsort()
importances = pd.DataFrame(perm.importances[sorted_importances_idx].T, 
                           columns=final[['Aspect', 'Lg_site_dist', 'PRCP', 'Slope', 'Stream_dist', 'TRI']].columns[sorted_importances_idx])
print(f'Permutation Importances:\n{importances}')

#Plots from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].barh(mdi_df['Feature'], mdi_df['Importance'])
ax[0].set_title('Feature Importance\nMean Decrease in Impurity')
ax[1].boxplot(importances, orientation = 'horizontal')
ax[1].set_title(f"Permutation Importances (test set)\nn = {n_repeats}")
ax[1].axvline(x=0, color="cornflowerblue", linestyle="--")
ax[1].set_xlabel("Decrease in accuracy score")
ax[1].set_yticklabels(importances.columns)
# ax[1] = importances.plot.box(vert=False, whis=10)
# ax[1].set_title("Permutation Importances (test set)")
# ax[1].axvline(x=0, color="k", linestyle="--")
# ax[1].set_xlabel("Decrease in accuracy score")
fig.tight_layout()
plt.savefig('figs/Figure_3.jpg', dpi = 300, bbox_inches = 'tight')
