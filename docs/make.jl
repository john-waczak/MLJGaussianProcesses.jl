using MLJGP
using Documenter

DocMeta.setdocmeta!(MLJGP, :DocTestSetup, :(using MLJGP); recursive=true)

makedocs(;
    modules=[MLJGP],
    authors="John Waczak <john.louis.waczak@gmail.com>",
    repo="https://github.com/john-waczak/MLJGP.jl/blob/{commit}{path}#{line}",
    sitename="MLJGP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/MLJGP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/MLJGP.jl",
    devbranch="main",
)
