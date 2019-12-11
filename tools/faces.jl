# get all faces, including
#   bboxes, landmarks, features
using Glob
using PyCall
using JLD2
using FileIO

const FR = pyimport("evolveface")
const PIL = pyimport("PIL")

# get all templates
template_files = glob("data/*.jpg")
template_features = hcat(map(template_files) do f
    img = PIL.Image.open(f)
    bounding_boxes, landmarks = FR.detect_faces(img)
    features = FR.extract_feature_IR50A(img, landmarks)
    @assert size(features)[1] == 1
    return vec(features)
end...)

@save "out/template.jld2" template_features

map(glob("data/vid/*.jpg")) do f
    println(f)
    img = PIL.Image.open(f)
    bounding_boxes, landmarks = FR.detect_faces(img)
    size(landmarks)[1] == 0 && return
    features = FR.extract_feature_IR50A(img, landmarks)
    res = (bounding_boxes, landmarks, features)
    @save "out/$(splitext(basename(f))[1]).jld2" res
end
