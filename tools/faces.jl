# get all faces, including
#   bboxes, landmarks, features
using Plots
using Glob
using PyCall
using DataFrames
using Measures
using LinearAlgebra
using Distances
using Statistics
using JLD2
using FileIO

const FR = pyimport("evolveface")
const PIL = pyimport("PIL")

# get all templates
template_files = joinpath.(".", ["$i.jpg" for i in 0:9])
target_features = []
for f in template_files
    img = PIL.Image.open(f)
    bounding_boxes, landmarks = FR.detect_faces(img)
    features = FR.extract_feature_IR50A(img, landmarks)
    @assert size(features)[1] == 1
    push!(target_features, vec(features))
end
target_features = hcat(target_features...)
dists = filter(>(0), triu(pairwise(Euclidean(), target_features, dims = 2)))
threshold = mean(dists)
σ = std(dists)

dpi = 200
gr(size = (3, 4) .* dpi, dpi = dpi)

savefig("z.pdf")
savefig("z.png")
