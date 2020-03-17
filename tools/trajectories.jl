using JLD2
using FileIO
using Glob
using Distances
using Statistics

files = glob("out/img_*.jld2")
ids = map(files) do f
    parse(Int, match(r"img_(\d*)\.jld", f).captures[1])
end

templates = load("out/template.jld2", "template_features")
alldata = load.(files, "res")


x = pairwise(CosineDist(), templates, templates, dims = 2)

T = 0.6

y = filter(!isnothing, map(zip(ids, alldata)) do (id, (bboxes, landmarks, features))
    likely = vec(minimum(pairwise(CosineDist(), features', templates, dims = 2), dims = 2))
    n = argmin(likely)
    likely[n] > T && return nothing
    (id = id, p = likely[n], bbox = bboxes[n, :], landmark = landmarks[n,:], feature = features[n,:])
end)

@save "t.jld2" y
