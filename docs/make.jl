using MLJGaussianProcesses
using Documenter

DocMeta.setdocmeta!(MLJGaussianProcesses, :DocTestSetup, :(using MLJGaussianProcesses); recursive=true)

makedocs(;
    modules=[MLJGaussianProcesses],
    authors="John Waczak <john.louis.waczak@gmail.com>",
    repo="https://github.com/john-waczak/MLJGaussianProcesses.jl/blob/{commit}{path}#{line}",
    sitename="MLJGaussianProcesses.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/MLJGaussianProcesses.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/MLJGaussianProcesses.jl",
    devbranch="main",
)
