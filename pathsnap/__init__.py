"""PathSnap: map-matching noisy GPS tracks onto a road network.

Package layout
--------------
io          reading the raw .track / .nodes / .arcs / .route files
geo         projection to metres + point/segment geometry
network     road graph built from nodes/arcs (spatial index, adjacency, shortest paths)
labels      deriving a reference arc for every GPS observation from the .route file
candidates  candidate road segments for each GPS observation
features    feature engineering for (observation, candidate) pairs
viz         folium maps
metrics     classification + route-matching metrics
models      the model zoo used in the comparison
hmm         HMM / Viterbi decoder
inference   end-to-end prediction pipeline used by the API
"""

__version__ = "1.0.0"
