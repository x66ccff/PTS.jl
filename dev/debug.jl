# # julia --project=. dev/debug.jl

# # 1. Set up the environment and paths
# using Pkg
# Pkg.activate(@__DIR__)  # Activate the environment in the current directory

# # Add the local source path to LOAD_PATH
# push!(LOAD_PATH, joinpath(@__DIR__, ".."))  # Add the parent directory (project root) to the path
# # Ensure using the development version
# import Pkg
# Pkg.develop(path=joinpath(@__DIR__, ".."))  # Add this line to explicitly specify using the local version

# # 2. Load development tools
# using Revise

# # 3. Set up logging
# using Logging
# ENV["JULIA_DEBUG"] = "SymbolicRegression"
# debug_logger = ConsoleLogger(stderr, Logging.Debug)
# global_logger(debug_logger)

# # 4. Load the local package
# using SymbolicRegression

# # 5. Verify that the local version is being used
# @info "Package path:" pathof(SymbolicRegression)
# @assert contains(pathof(SymbolicRegression), pwd()) "Not using the local version!"

# ###################### 
# # exit()

# X = randn(Float32, 5, 100)
# # y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

# # y = 2 * cos.(X[4, :]).^3 + X[1, :] .^ 2 .- 2

# y = 2 * cos.(X[4, :]).^3 + X[1, :] .^ 2 .- 2

# options = SymbolicRegression.Options(;
#     binary_operators=[+, *, /, -], unary_operators=[cos, exp, sin, log]
# )

# # hall_of_fame = equation_search(X, y; options=options, parallelism=:multithreading)
# hall_of_fame = equation_search(X, y; options=options, parallelism=:serial)

# dominating = calculate_pareto_frontier(hall_of_fame)

# trees = [member.tree for member in dominating]

# tree = trees[end]
# output, did_succeed = eval_tree_array(tree, X, options)

# println("Complexity\tMSE\tEquation")

# for member in dominating
#     complexity = compute_complexity(member, options)
#     loss = member.loss
#     string = string_tree(member.tree, options)

#     println("$(complexity)\t$(loss)\t$(string)")
# end

###################################################################

# julia --project=. dev/debug.jl

# 1. Set up the environment and paths
using Pkg
Pkg.activate(@__DIR__)  # Activate the environment in the current directory

# Add StatProfilerHTML for profiling
using StatProfilerHTML

# Add the local source path to LOAD_PATH
push!(LOAD_PATH, joinpath(@__DIR__, ".."))  # Add the parent directory (project root) to the path
# Ensure using the development version
using Pkg: Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))  # Add this line to explicitly specify using the local version

# 2. Load development tools
using Revise

# 3. Set up logging
using Logging
ENV["JULIA_DEBUG"] = "SymbolicRegression"
debug_logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(debug_logger)

# 4. Load the local package
using SymbolicRegression

# 5. Verify that the local version is being used
@info "Package path:" pathof(SymbolicRegression)
@assert contains(pathof(SymbolicRegression), pwd()) "Not using the local version!"

###################### 
# exit()

# 包装主要计算逻辑到一个函数中
function main_computation()
    X = randn(Float32, 5, 100)
    y = 2 * cos.(X[4, :]) .^ 3 + X[1, :] .^ 2 .- 2

    options = SymbolicRegression.Options(;
        binary_operators=[+, *, /, -], unary_operators=[cos, exp, sin, log]
    )

    # hall_of_fame = equation_search(X, y; options=options, parallelism=:multithreading)
    hall_of_fame = equation_search(X, y; options=options, parallelism=:serial)
    dominating = calculate_pareto_frontier(hall_of_fame)
    trees = [member.tree for member in dominating]
    tree = trees[end]
    output, did_succeed = eval_tree_array(tree, X, options)

    println("Complexity\tMSE\tEquation")
    for member in dominating
        complexity = compute_complexity(member, options)
        loss = member.loss
        string = string_tree(member.tree, options)
        println("$(complexity)\t$(loss)\t$(string)")
    end
end

# 使用StatProfilerHTML进行性能分析
# @profilehtml main_computation()
main_computation()
