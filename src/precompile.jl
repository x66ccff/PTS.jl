using PrecompileTools: @compile_workload, @setup_workload
using CUDA
macro maybe_setup_workload(mode, ex)
    precompile_ex = Expr(
        :macrocall, Symbol("@setup_workload"), LineNumberNode(@__LINE__), ex
    )
    return quote
        if $(esc(mode)) == :compile
            $(esc(ex))
        elseif $(esc(mode)) == :precompile
            $(esc(precompile_ex))
        else
            error("Invalid value for mode: " * show($(esc(mode))))
        end
    end
end

macro maybe_compile_workload(mode, ex)
    precompile_ex = Expr(
        :macrocall, Symbol("@compile_workload"), LineNumberNode(@__LINE__), ex
    )
    return quote
        if $(esc(mode)) == :compile
            $(esc(ex))
        elseif $(esc(mode)) == :precompile
            $(esc(precompile_ex))
        else
            error("Invalid value for mode: " * show($(esc(mode))))
        end
    end
end

function _precompile_psrn_evaluation()
    @setup_workload begin
        T = Float32
        n_samples = 10
        n_subtrees = 5

        X_mapped = rand(T, n_samples, n_subtrees)
        y = rand(T, 1, n_samples)

        options = Options(;
            binary_operators=(+, *, -, /), unary_operators=(cos, sin, exp, log)
        )
        operators = options.operators
        variable_names = ["x1"]

        trees = Vector{Expression}()
        for i in 1:n_subtrees
            x1 = Expression(Node{T}(; feature=1); operators, variable_names)
            tree = x1 * x1
            push!(trees, tree)
        end
        @compile_workload begin
            psrn = PSRN(;
                n_variables=n_subtrees,
                operators=[
                    "Add", "Mul", "Sub", "Div", "Identity", "Cos", "Sin", "Exp", "Log"
                ],
                n_symbol_layers=2,
                backend=CUDA.CUDABackend(),
                initial_expressions=trees,
                options=options,
            )

            best_expressions, mse_values = get_best_expressions(
                psrn, X_mapped, y; top_k=100
            )
        end
    end
end

"""`mode=:precompile` will use `@precompile_*` directives; `mode=:compile` runs."""
function do_precompilation(::Val{mode}) where {mode}
    @maybe_setup_workload mode begin
        for T in [Float32, Float64], nout in 1:2
            start = nout == 1
            N = 30
            X = randn(T, 3, N)
            y = start ? randn(T, N) : randn(T, nout, N)
            @maybe_compile_workload mode begin
                options = SymbolicRegression.Options(;
                    binary_operators=[+, *, /, -, ^],
                    unary_operators=[sin, cos, exp, log, sqrt, abs],
                    populations=3,
                    population_size=start ? 50 : 12,
                    tournament_selection_n=6,
                    ncycles_per_iteration=start ? 30 : 1,
                    mutation_weights=MutationWeights(;
                        mutate_constant=1.0,
                        mutate_operator=1.0,
                        swap_operands=1.0,
                        add_node=1.0,
                        insert_node=1.0,
                        delete_node=1.0,
                        simplify=1.0,
                        randomize=1.0,
                        do_nothing=1.0,
                        optimize=1.0,
                    ),
                    fraction_replaced=0.2,
                    fraction_replaced_hof=0.2,
                    define_helper_functions=false,
                    optimizer_probability=0.05,
                    save_to_file=false,
                )
                state = equation_search(
                    X,
                    y;
                    niterations=start ? 3 : 1,
                    options=options,
                    parallelism=:multithreading,
                    return_state=true,
                    verbosity=0,
                )
                hof = equation_search(
                    X,
                    y;
                    niterations=0,
                    options=options,
                    parallelism=:multithreading,
                    saved_state=state,
                    return_state=false,
                    verbosity=0,
                )
                nout == 1 && calculate_pareto_frontier(hof::HallOfFame)
            end
        end
    end

    precompile(PSRN, (Int, Vector{String}, Int, Any, Vector{Expression}, Options))
    precompile(get_best_expressions, (PSRN, AbstractArray, AbstractArray))

    _precompile_psrn_evaluation()

    return nothing
end
