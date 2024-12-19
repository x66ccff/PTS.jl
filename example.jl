# julia --project=. example.jl

using SymbolicRegression

function main()
    X = randn(Float32, 5, 100)
    # X = randn(Float64, 5, 101) # test for Float64, passed
    # X = randn(Float64, 5, 10100) # test for PSRN downsampling, passed

    # y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2
    y = 2 * cos.(X[4, :]) .^ 3 + X[1, :] .^ 2 .- 2 # harder problem

    options = SymbolicRegression.Options(;
        binary_operators=[+, *, /, -], unary_operators=[cos, exp, log, sin]
    )

    hall_of_fame = equation_search(X, y; options=options, parallelism=:multithreading)

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

@time main()
