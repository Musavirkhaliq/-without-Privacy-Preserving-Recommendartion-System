using ArgParse, DelimitedFiles, LinearAlgebra, Random

struct Tile
    min_x::Int
    max_x::Int
    min_y::Int
    max_y::Int
end

are_itersecting(s::Tile, t::Tile) = !(s.min_x > t.max_x || s.max_x < t.min_x || s.min_y > t.max_y || s.max_y < t.min_y)
add_noise(x, pflip) = eltype(x)((x != 0) | (rand() <= pflip))
perturb(x, pflip) = eltype(x)((x != 0) ⊻ (rand() <= pflip))

function random_tile(n, m, min_width, max_width, min_height, max_height)
    x = rand(1:n)
    y = rand(1:m)
    w = rand(min_width:max_width)
    h = rand(min_height:max_height)
    Tile(x, min(n, x + w), y, min(m, y + h))
end

function generate_tiles(
    n::Int,
    m::Int,
    num_tiles::Int,
    tile_width_range::Tuple{Int,Int},
    tile_height_range::Tuple{Int,Int};
    max_trials = 10,
    allow_overlap = true,
)
    tiles = Tile[]
    trials = 0
    while length(tiles) < num_tiles && trials < max_trials
        t = random_tile(n - 1, m - 1, tile_width_range[1], tile_width_range[2], tile_height_range[1], tile_height_range[2])
        if allow_overlap || !any(s -> are_itersecting(s, t), tiles)
            push!(tiles, t)
            trials = 0
        else
            trials = trials + 1
        end
    end
    tiles
end

function generate_tiled_random_matrix(n::Int, m::Int, k::Int, area_boundaries::Tuple{Int,Int}, noise::Float64; allow_overlap = true)
    tiles = generate_tiles(n, m, k, area_boundaries, area_boundaries; allow_overlap = allow_overlap)
    A = zeros(Int8, (n, m))
    for t in tiles
        view(A, t.min_x:t.max_x, t.min_y:t.max_y) .= 1
    end
    add_noise.(A, noise)
end


function main()
    s = ArgParseSettings(description = "Elbmf CLI")
    @add_arg_table! s begin
        "-o", "--output"
            help = "output matrix as tsv"
            required = true
        "-n", "--height"
            arg_type = Int
            required = true
            help = "The height of the random matrix"
        "-m", "--width"
            arg_type = Int
            required = true
            help = "The width of the random matrix"
        "-k", "--numtiles"
            arg_type = Int
            required = true
            help = "The number of tiles"
        "--minwidth"
            arg_type = Int
            required = true
            help = "The min width of a tile"
        "--maxwidth"
            arg_type = Int
            required = true
            help = "The max width of a tile"
        "--seed"
            arg_type = Int
            required = false
            help = "The random seed"
        "--allowoverlap"
            action = :store_true
            help = "Allow tiles to overlap"
        "--additivenoise"
            arg_type = Float32
            required = true
            help = "The level of additive noise"

    end
    opts = parse_args(s)
    
    if !isnothing(opts["seed"])
        Random.seed!(opts["seed"])
    end

    A = Elbmf.generate_tiled_random_matrix(
        opts["n"],
        opts["m"],
        opts["k"],
        (opts["minwidth"], opts["maxwidth"]),
        opts["additivenoise"];
        allow_overlap = opts["allowoverlap"]
    )
    writedlm(opts["output"], A, '\t')
end

main()
