module SymbolicRegression

# Types
export Population,
    PopMember,
    HallOfFame,
    Options,
    Dataset,
    MutationWeights,
    Node,
    GraphNode,
    ParametricNode,
    Expression,
    ParametricExpression,
    TemplateExpression,
    TemplateStructure,
    ValidVector,
    ComposableExpression,
    NodeSampler,
    AbstractExpression,
    AbstractExpressionNode,
    EvalOptions,
    SRRegressor,
    MultitargetSRRegressor,
    SRLogger,

    #Functions:
    equation_search,
    s_r_cycle,
    calculate_pareto_frontier,
    count_nodes,
    compute_complexity,
    @parse_expression,
    parse_expression,
    @declare_expression_operator,
    print_tree,
    string_tree,
    eval_tree_array,
    eval_diff_tree_array,
    eval_grad_tree_array,
    differentiable_eval_tree_array,
    set_node!,
    copy_node,
    node_to_symbolic,
    symbolic_to_node,
    simplify_tree!,
    tree_mapreduce,
    combine_operators,
    gen_random_tree,
    gen_random_tree_fixed_size,
    @extend_operators,
    get_tree,
    get_contents,
    get_metadata,
    with_contents,
    with_metadata,

    #Operators
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_acosh,
    safe_sqrt,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,

    # special operators
    gamma,
    erf,
    erfc,
    atanh_clip,

    PSRNManager,
    start_psrn_task,
    process_psrn_results!

using Distributed
using Printf: @printf, @sprintf
using Pkg: Pkg
using TOML: parsefile
using Random: seed!, shuffle!, randperm
using Reexport
using ProgressMeter: finish!
using DynamicExpressions:
    Node,
    GraphNode,
    ParametricNode,
    Expression,
    ParametricExpression,
    NodeSampler,
    AbstractExpression,
    AbstractExpressionNode,
    ExpressionInterface,
    @parse_expression,
    parse_expression,
    @declare_expression_operator,
    copy_node,
    set_node!,
    string_tree,
    print_tree,
    count_nodes,
    get_constants,
    get_scalar_constants,
    set_constants!,
    set_scalar_constants!,
    index_constants,
    NodeIndex,
    eval_tree_array,
    EvalOptions,
    differentiable_eval_tree_array,
    eval_diff_tree_array,
    eval_grad_tree_array,
    node_to_symbolic,
    symbolic_to_node,
    combine_operators,
    simplify_tree!,
    tree_mapreduce,
    set_default_variable_names!,
    node_type,
    get_tree,
    get_contents,
    get_metadata,
    with_contents,
    with_metadata
using DynamicExpressions: with_type_parameters
@reexport using LossFunctions:
    MarginLoss,
    DistanceLoss,
    SupervisedLoss,
    ZeroOneLoss,
    LogitMarginLoss,
    PerceptronLoss,
    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,
    L2MarginLoss,
    ExpLoss,
    SigmoidLoss,
    DWDMarginLoss,
    LPDistLoss,
    L1DistLoss,
    L2DistLoss,
    PeriodicLoss,
    HuberLoss,
    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,
    LogitDistLoss,
    QuantileLoss,
    LogCoshLoss
using DynamicDiff: D
using Compat: @compat, Fix
using CUDA
using KernelAbstractions

#! format: off
@compat(
    public,
    (
        AbstractOptions, AbstractRuntimeOptions, RuntimeOptions,
        AbstractMutationWeights, mutate!, condition_mutation_weights!,
        sample_mutation, MutationResult, AbstractSearchState, SearchState,
        LOSS_TYPE, DATA_TYPE, node_type,
    )
)
#! format: on
# ^ We can add new functions here based on requests from users.
# However, I don't want to add many functions without knowing what
# users will actually want to overload.

# https://discourse.julialang.org/t/how-to-find-out-the-version-of-a-package-from-its-module/37755/15
const PACKAGE_VERSION = try
    root = pkgdir(@__MODULE__)
    if root == String
        let project = parsefile(joinpath(root, "Project.toml"))
            VersionNumber(project["version"])
        end
    else
        VersionNumber(0, 0, 0)
    end
catch
    VersionNumber(0, 0, 0)
end

using DispatchDoctor: @stable

@stable default_mode = "disable" begin
    include("Utils.jl")
    include("InterfaceDynamicQuantities.jl")
    include("Core.jl")
    include("InterfaceDynamicExpressions.jl")
    include("Recorder.jl")
    include("Complexity.jl")
    include("DimensionalAnalysis.jl")
    include("CheckConstraints.jl")
    include("AdaptiveParsimony.jl")
    include("MutationFunctions.jl")
    include("LossFunctions.jl")
    include("PopMember.jl")
    include("ConstantOptimization.jl")
    include("Population.jl")
    include("HallOfFame.jl")
    include("Mutate.jl")
    include("RegularizedEvolution.jl")
    include("SingleIteration.jl")
    include("ProgressBars.jl")
    include("Migration.jl")
    include("SearchUtils.jl")
    include("Logging.jl")
    include("ExpressionBuilder.jl")
    include("ComposableExpression.jl")
    include("TemplateExpression.jl")
    include("ParametricExpression.jl")
    include("PSRNfunctions.jl")
    include("PSRNmodel.jl")
end

using .CoreModule:
    DATA_TYPE,
    LOSS_TYPE,
    RecordType,
    Dataset,
    AbstractOptions,
    Options,
    ComplexityMapping,
    AbstractMutationWeights,
    MutationWeights,
    get_safe_op,
    max_features,
    is_weighted,
    sample_mutation,
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip,
    create_expression,
    has_units
using .UtilsModule: is_anonymous_function, recursive_merge, json3_write, @ignore
using .ComplexityModule: compute_complexity
using .CheckConstraintsModule: check_constraints
using .AdaptiveParsimonyModule:
    RunningSearchStatistics, update_frequencies!, move_window!, normalize_frequencies!
using .MutationFunctionsModule:
    gen_random_tree,
    gen_random_tree_fixed_size,
    random_node,
    random_node_and_parent,
    crossover_trees
using .InterfaceDynamicExpressionsModule: @extend_operators
using .LossFunctionsModule: eval_loss, score_func, update_baseline_loss!
using .PopMemberModule: PopMember, reset_birth!
using .PopulationModule: Population, best_sub_pop, record_population, best_of_sample
using .HallOfFameModule:
    HallOfFame, calculate_pareto_frontier, string_dominating_pareto_curve
using .MutateModule: mutate!, condition_mutation_weights!, MutationResult
using .SingleIterationModule: s_r_cycle, optimize_and_simplify_population
using .ProgressBarsModule: WrappedProgressBar
using .RecorderModule: @recorder, find_iteration_from_record
using .MigrationModule: migrate!
using .SearchUtilsModule:
    AbstractSearchState,
    SearchState,
    AbstractRuntimeOptions,
    RuntimeOptions,
    WorkerAssignments,
    DefaultWorkerOutputType,
    assign_next_worker!,
    get_worker_output_type,
    extract_from_worker,
    @sr_spawner,
    StdinReader,
    watch_stream,
    close_reader!,
    check_for_user_quit,
    check_for_loss_threshold,
    check_for_timeout,
    check_max_evals,
    ResourceMonitor,
    record_channel_state!,
    estimate_work_fraction,
    update_progress_bar!,
    print_search_state,
    init_dummy_pops,
    load_saved_hall_of_fame,
    load_saved_population,
    construct_datasets,
    save_to_file,
    get_cur_maxsize,
    update_hall_of_fame!,
    logging_callback!
using .LoggingModule: AbstractSRLogger, SRLogger, get_logger
using .TemplateExpressionModule: TemplateExpression, TemplateStructure
using .TemplateExpressionModule: TemplateExpression, TemplateStructure, ValidVector
using .ComposableExpressionModule: ComposableExpression
using .ExpressionBuilderModule: embed_metadata, strip_metadata

import .PSRNmodel: PSRN, forward, get_expr, get_best_expressions



@stable default_mode = "disable" begin
    include("deprecates.jl")
    include("Configure.jl")
end

using KernelAbstractions
using CUDA
@static if Base.find_package("Metal") !== nothing
    using Metal
end
@static if Base.find_package("AMDGPU") !== nothing
    using AMDGPU
end
@static if Base.find_package("oneAPI") !== nothing
    using oneAPI
end

const USE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

const AVAILABLE_BACKENDS = Dict{Symbol,Any}()

function __init__()
    if USE_CUDA
        AVAILABLE_BACKENDS[:cuda] = CUDABackend()
    end
    if @isdefined(Metal) && Metal.functional()
        AVAILABLE_BACKENDS[:metal] = MetalBackend()
    end
    if @isdefined(AMDGPU) && AMDGPU.functional()
        AVAILABLE_BACKENDS[:rocm] = ROCBackend()
    end
    if @isdefined(oneAPI) && oneAPI.functional()
        AVAILABLE_BACKENDS[:oneapi] = oneBackend()
    end
    AVAILABLE_BACKENDS[:cpu] = CPU()
end

function get_preferred_backend()
    for backend in [:cuda, :metal, :rocm, :oneapi, :cpu]
        if haskey(AVAILABLE_BACKENDS, backend)
            return AVAILABLE_BACKENDS[backend]
        end
    end
    return AVAILABLE_BACKENDS[:cpu]
end

"""
    equation_search(X, y[; kws...])

Perform a distributed equation search for functions `f_i` which
describe the mapping `f_i(X[:, j]) ≈ y[i, j]`. Options are
configured using SymbolicRegression.Options(...),
which should be passed as a keyword argument to options.
One can turn off parallelism with `numprocs=0`,
which is useful for debugging and profiling.

# Arguments
- `X::AbstractMatrix{T}`:  The input dataset to predict `y` from.
    The first dimension is features, the second dimension is rows.
- `y::Union{AbstractMatrix{T}, AbstractVector{T}}`: The values to predict. The first dimension
    is the output feature to predict with each equation, and the
    second dimension is rows.
- `niterations::Int=100`: The number of iterations to perform the search.
    More iterations will improve the results.
- `weights::Union{AbstractMatrix{T}, AbstractVector{T}, Nothing}=nothing`: Optionally
    weight the loss for each `y` by this value (same shape as `y`).
- `options::AbstractOptions=Options()`: The options for the search, such as
    which operators to use, evolution hyperparameters, etc.
- `variable_names::Union{Vector{String}, Nothing}=nothing`: The names
    of each feature in `X`, which will be used during printing of equations.
- `display_variable_names::Union{Vector{String}, Nothing}=variable_names`: Names
    to use when printing expressions during the search, but not when saving
    to an equation file.
- `y_variable_names::Union{String,AbstractVector{String},Nothing}=nothing`: The
    names of each output feature in `y`, which will be used during printing
    of equations.
- `parallelism=:multithreading`: What parallelism mode to use.
    The options are `:multithreading`, `:multiprocessing`, and `:serial`.
    By default, multithreading will be used. Multithreading uses less memory,
    but multiprocessing can handle multi-node compute. If using `:multithreading`
    mode, the number of threads available to julia are used. If using
    `:multiprocessing`, `numprocs` processes will be created dynamically if
    `procs` is unset. If you have already allocated processes, pass them
    to the `procs` argument and they will be used.
    You may also pass a string instead of a symbol, like `"multithreading"`.
- `numprocs::Union{Int, Nothing}=nothing`:  The number of processes to use,
    if you want `equation_search` to set this up automatically. By default
    this will be `4`, but can be any number (you should pick a number <=
    the number of cores available).
- `procs::Union{Vector{Int}, Nothing}=nothing`: If you have set up
    a distributed run manually with `procs = addprocs()` and `@everywhere`,
    pass the `procs` to this keyword argument.
- `addprocs_function::Union{Function, Nothing}=nothing`: If using multiprocessing
    (`parallelism=:multiprocessing`), and are not passing `procs` manually,
    then they will be allocated dynamically using `addprocs`. However,
    you may also pass a custom function to use instead of `addprocs`.
    This function should take a single positional argument,
    which is the number of processes to use, as well as the `lazy` keyword argument.
    For example, if set up on a slurm cluster, you could pass
    `addprocs_function = addprocs_slurm`, which will set up slurm processes.
- `heap_size_hint_in_bytes::Union{Int,Nothing}=nothing`: On Julia 1.9+, you may set the `--heap-size-hint`
    flag on Julia processes, recommending garbage collection once a process
    is close to the recommended size. This is important for long-running distributed
    jobs where each process has an independent memory, and can help avoid
    out-of-memory errors. By default, this is set to `Sys.free_memory() / numprocs`.
- `worker_imports::Union{Vector{Symbol},Nothing}=nothing`: If you want to import
    additional modules on each worker, pass them here as a vector of symbols.
    By default some of the extensions will automatically be loaded when needed.
- `runtests::Bool=true`: Whether to run (quick) tests before starting the
    search, to see if there will be any problems during the equation search
    related to the host environment.
- `saved_state=nothing`: If you have already
    run `equation_search` and want to resume it, pass the state here.
    To get this to work, you need to have set return_state=true,
    which will cause `equation_search` to return the state. The second
    element of the state is the regular return value with the hall of fame.
    Note that you cannot change the operators or dataset, but most other options
    should be changeable.
- `return_state::Union{Bool, Nothing}=nothing`: Whether to return the
    state of the search for warm starts. By default this is false.
- `loss_type::Type=Nothing`: If you would like to use a different type
    for the loss than for the data you passed, specify the type here.
    Note that if you pass complex data `::Complex{L}`, then the loss
    type will automatically be set to `L`.
- `verbosity`: Whether to print debugging statements or not.
- `logger::Union{AbstractSRLogger,Nothing}=nothing`: An optional logger to record
    the progress of the search. You can use an `SRLogger` to wrap a custom logger,
    or pass `nothing` to disable logging.
- `progress`: Whether to use a progress bar output. Only available for
    single target output.
- `X_units::Union{AbstractVector,Nothing}=nothing`: The units of the dataset,
    to be used for dimensional constraints. For example, if `X_units=["kg", "m"]`,
    then the first feature will have units of kilograms, and the second will
    have units of meters.
- `y_units=nothing`: The units of the output, to be used for dimensional constraints.
    If `y` is a matrix, then this can be a vector of units, in which case
    each element corresponds to each output feature.

# Returns
- `hallOfFame::HallOfFame`: The best equations seen during the search.
    hallOfFame.members gives an array of `PopMember` objects, which
    have their tree (equation) stored in `.tree`. Their score (loss)
    is given in `.score`. The array of `PopMember` objects
    is enumerated by size from `1` to `options.maxsize`.
"""
function equation_search(
    X::AbstractMatrix{T},
    y::AbstractMatrix;
    niterations::Int=100,
    weights::Union{AbstractMatrix{T},AbstractVector{T},Nothing}=nothing,
    options::AbstractOptions=Options(),
    variable_names::Union{AbstractVector{String},Nothing}=nothing,
    display_variable_names::Union{AbstractVector{String},Nothing}=variable_names,
    y_variable_names::Union{String,AbstractVector{String},Nothing}=nothing,
    parallelism=:multithreading,
    numprocs::Union{Int,Nothing}=nothing,
    procs::Union{Vector{Int},Nothing}=nothing,
    addprocs_function::Union{Function,Nothing}=nothing,
    heap_size_hint_in_bytes::Union{Integer,Nothing}=nothing,
    worker_imports::Union{Vector{Symbol},Nothing}=nothing,
    runtests::Bool=true,
    saved_state=nothing,
    return_state::Union{Bool,Nothing,Val}=nothing,
    run_id::Union{String,Nothing}=nothing,
    loss_type::Type{L}=Nothing,
    verbosity::Union{Integer,Nothing}=nothing,
    logger::Union{AbstractSRLogger,Nothing}=nothing,
    progress::Union{Bool,Nothing}=nothing,
    X_units::Union{AbstractVector,Nothing}=nothing,
    y_units=nothing,
    extra::NamedTuple=NamedTuple(),
    v_dim_out::Val{DIM_OUT}=Val(nothing),
    # Deprecated:
    multithreaded=nothing,
) where {T<:DATA_TYPE,L,DIM_OUT}
    if multithreaded !== nothing
        error(
            "`multithreaded` is deprecated. Use the `parallelism` argument instead. " *
            "Choose one of :multithreaded, :multiprocessing, or :serial.",
        )
    end

    if weights !== nothing
        @assert length(weights) == length(y)
        weights = reshape(weights, size(y))
    end

    datasets = construct_datasets(
        X,
        y,
        weights,
        variable_names,
        display_variable_names,
        y_variable_names,
        X_units,
        y_units,
        extra,
        L,
    )

    return equation_search(
        datasets;
        niterations=niterations,
        options=options,
        parallelism=parallelism,
        numprocs=numprocs,
        procs=procs,
        addprocs_function=addprocs_function,
        heap_size_hint_in_bytes=heap_size_hint_in_bytes,
        worker_imports=worker_imports,
        runtests=runtests,
        saved_state=saved_state,
        return_state=return_state,
        run_id=run_id,
        verbosity=verbosity,
        logger=logger,
        progress=progress,
        v_dim_out=Val(DIM_OUT),
    )
end

function equation_search(
    X::AbstractMatrix{T}, y::AbstractVector; kw...
) where {T<:DATA_TYPE}
    return equation_search(X, reshape(y, (1, size(y, 1))); kw..., v_dim_out=Val(1))
end

function equation_search(dataset::Dataset; kws...)
    return equation_search([dataset]; kws..., v_dim_out=Val(1))
end

function equation_search(
    datasets::Vector{D};
    options::AbstractOptions=Options(),
    saved_state=nothing,
    runtime_options::Union{AbstractRuntimeOptions,Nothing}=nothing,
    runtime_options_kws...,
) where {T<:DATA_TYPE,L<:LOSS_TYPE,D<:Dataset{T,L}}
    _runtime_options = @something(
        runtime_options,
        RuntimeOptions(;
            options_return_state=options.return_state,
            options_verbosity=options.verbosity,
            options_progress=options.progress,
            nout=length(datasets),
            runtime_options_kws...,
        )
    )

    # Underscores here mean that we have mutated the variable
    return _equation_search(datasets, _runtime_options, options, saved_state)
end

@noinline function _equation_search(
    datasets::Vector{D}, ropt::AbstractRuntimeOptions, options::AbstractOptions, saved_state
) where {D<:Dataset}
    _validate_options(datasets, ropt, options)
    state = _create_workers(datasets, ropt, options)
    _initialize_search!(state, datasets, ropt, options, saved_state)
    _warmup_search!(state, datasets, ropt, options)
    _main_search_loop!(state, datasets, ropt, options)
    _tear_down!(state, ropt, options)
    _info_dump(state, datasets, ropt, options)
    return _format_output(state, datasets, ropt, options)
end

function _validate_options(
    datasets::Vector{D}, ropt::AbstractRuntimeOptions, options::AbstractOptions
) where {T,L,D<:Dataset{T,L}}
    example_dataset = first(datasets)
    nout = length(datasets)
    @assert nout >= 1
    @assert (nout == 1 || ropt.dim_out == 2)
    @assert options.populations >= 1
    if ropt.progress
        @assert(nout == 1, "You cannot display a progress bar for multi-output searches.")
        @assert(ropt.verbosity > 0, "You cannot display a progress bar with `verbosity=0`.")
    end
    if options.node_type <: GraphNode && ropt.verbosity > 0
        @warn "The `GraphNode` interface and mutation operators are experimental and will change in future versions."
    end
    if ropt.runtests
        test_option_configuration(ropt.parallelism, datasets, options, ropt.verbosity)
        test_dataset_configuration(example_dataset, options, ropt.verbosity)
    end
    for dataset in datasets
        update_baseline_loss!(dataset, options)
    end
    if options.define_helper_functions
        set_default_variable_names!(first(datasets).variable_names)
    end
    if options.seed !== nothing
        seed!(options.seed)
    end
    return nothing
end
@stable default_mode = "disable" function _create_workers(
    datasets::Vector{D}, ropt::AbstractRuntimeOptions, options::AbstractOptions
) where {T,L,D<:Dataset{T,L}}
    stdin_reader = watch_stream(options.input_stream)

    record = RecordType()
    @recorder record["options"] = "$(options)"

    nout = length(datasets)
    example_dataset = first(datasets)
    example_ex = create_expression(zero(T), options, example_dataset)
    NT = typeof(example_ex)
    PopType = Population{T,L,NT}
    HallOfFameType = HallOfFame{T,L,NT}
    WorkerOutputType = get_worker_output_type(
        Val(ropt.parallelism), PopType, HallOfFameType
    )
    ChannelType = ropt.parallelism == :multiprocessing ? RemoteChannel : Channel

    # Pointers to populations on each worker:
    worker_output = Vector{WorkerOutputType}[WorkerOutputType[] for j in 1:nout]
    # Initialize storage for workers
    tasks = [Task[] for j in 1:nout]
    # Set up a channel to send finished populations back to head node
    channels = [[ChannelType(1) for i in 1:(options.populations)] for j in 1:nout]
    (procs, we_created_procs) = if ropt.parallelism == :multiprocessing
        configure_workers(;
            procs=ropt.init_procs,
            ropt.numprocs,
            ropt.addprocs_function,
            options,
            worker_imports=ropt.worker_imports,
            project_path=splitdir(Pkg.project().path)[1],
            file=@__FILE__,
            ropt.exeflags,
            ropt.verbosity,
            example_dataset,
            ropt.runtests,
        )
    else
        Int[], false
    end
    # Get the next worker process to give a job:
    worker_assignment = WorkerAssignments()
    # Randomly order which order to check populations:
    # This is done so that we do work on all nout equally.
    task_order = [(j, i) for j in 1:nout for i in 1:(options.populations)]
    shuffle!(task_order)

    # Persistent storage of last-saved population for final return:
    last_pops = init_dummy_pops(options.populations, datasets, options)
    # Best 10 members from each population for migration:
    best_sub_pops = init_dummy_pops(options.populations, datasets, options)
    # TODO: Should really be one per population too.
    all_running_search_statistics = [
        RunningSearchStatistics(; options=options) for j in 1:nout
    ]
    # Records the number of evaluations:
    # Real numbers indicate use of batching.
    num_evals = [[0.0 for i in 1:(options.populations)] for j in 1:nout]

    halls_of_fame = Vector{HallOfFameType}(undef, nout)

    total_cycles = ropt.niterations * options.populations
    cycles_remaining = [total_cycles for j in 1:nout]
    cur_maxsizes = [
        get_cur_maxsize(; options, total_cycles, cycles_remaining=cycles_remaining[j]) for
        j in 1:nout
    ]

    return SearchState{T,L,typeof(example_ex),WorkerOutputType,ChannelType}(;
        procs=procs,
        we_created_procs=we_created_procs,
        worker_output=worker_output,
        tasks=tasks,
        channels=channels,
        worker_assignment=worker_assignment,
        task_order=task_order,
        halls_of_fame=halls_of_fame,
        last_pops=last_pops,
        best_sub_pops=best_sub_pops,
        all_running_search_statistics=all_running_search_statistics,
        num_evals=num_evals,
        cycles_remaining=cycles_remaining,
        cur_maxsizes=cur_maxsizes,
        stdin_reader=stdin_reader,
        record=Ref(record),
    )
end
function _initialize_search!(
    state::AbstractSearchState{T,L,N},
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
    saved_state,
) where {T,L,N}
    nout = length(datasets)

    init_hall_of_fame = load_saved_hall_of_fame(saved_state)
    if init_hall_of_fame === nothing
        for j in 1:nout
            state.halls_of_fame[j] = HallOfFame(options, datasets[j])
        end
    else
        # Recompute losses for the hall of fame, in
        # case the dataset changed:
        for j in eachindex(init_hall_of_fame, datasets, state.halls_of_fame)
            hof = strip_metadata(init_hall_of_fame[j], options, datasets[j])
            for member in hof.members[hof.exists]
                score, result_loss = score_func(datasets[j], member, options)
                member.score = score
                member.loss = result_loss
            end
            state.halls_of_fame[j] = hof
        end
    end

    for j in 1:nout, i in 1:(options.populations)
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )
        saved_pop = load_saved_population(saved_state; out=j, pop=i)
        new_pop =
            if saved_pop !== nothing && length(saved_pop.members) == options.population_size
                _saved_pop = strip_metadata(saved_pop, options, datasets[j])
                ## Update losses:
                for member in _saved_pop.members
                    score, result_loss = score_func(datasets[j], member, options)
                    member.score = score
                    member.loss = result_loss
                end
                copy_pop = copy(_saved_pop)
                @sr_spawner(
                    begin
                        (copy_pop, HallOfFame(options, datasets[j]), RecordType(), 0.0)
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
            else
                if saved_pop !== nothing && ropt.verbosity > 0
                    @warn "Recreating population (output=$(j), population=$(i)), as the saved one doesn't have the correct number of members."
                end
                @sr_spawner(
                    begin
                        (
                            Population(
                                datasets[j];
                                population_size=options.population_size,
                                nlength=3,
                                options=options,
                                nfeatures=max_features(datasets[j], options),
                            ),
                            HallOfFame(options, datasets[j]),
                            RecordType(),
                            Float64(options.population_size),
                        )
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
                # This involves population_size evaluations, on the full dataset:
            end
        push!(state.worker_output[j], new_pop)
    end
    return nothing
end
function _warmup_search!(
    state::AbstractSearchState{T,L,N},
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
) where {T,L,N}
    nout = length(datasets)
    for j in 1:nout, i in 1:(options.populations)
        dataset = datasets[j]
        running_search_statistics = state.all_running_search_statistics[j]
        cur_maxsize = state.cur_maxsizes[j]
        @recorder state.record[]["out$(j)_pop$(i)"] = RecordType()
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )

        # TODO - why is this needed??
        # Multi-threaded doesn't like to fetch within a new task:
        c_rss = deepcopy(running_search_statistics)
        last_pop = state.worker_output[j][i]
        updated_pop = @sr_spawner(
            begin
                in_pop = first(
                    extract_from_worker(last_pop, Population{T,L,N}, HallOfFame{T,L,N})
                )
                _dispatch_s_r_cycle(
                    in_pop,
                    dataset,
                    options;
                    pop=i,
                    out=j,
                    iteration=0,
                    ropt.verbosity,
                    cur_maxsize,
                    running_search_statistics=c_rss,
                )::DefaultWorkerOutputType{Population{T,L,N},HallOfFame{T,L,N}}
            end,
            parallelism = ropt.parallelism,
            worker_idx = worker_idx
        )
        state.worker_output[j][i] = updated_pop
    end
    return nothing
end


mutable struct PSRNManager
    channel::Channel{Vector{Expression}}
    current_task::Union{Task,Nothing}
    call_count::Int
    N_PSRN_INPUT::Int
    net::PSRN
    max_samples::Int
    
    function PSRNManager(;
        N_PSRN_INPUT::Int,
        operators::Vector{String},
        n_symbol_layers::Int,
        options::Options,
        max_samples::Int=100
    )
        psrn = PSRN(
            n_variables = N_PSRN_INPUT,
            operators = operators,
            n_symbol_layers = n_symbol_layers,
            backend = CUDA.CUDABackend(),
            options = options
        )
        
        new(
            Channel{Vector{Expression}}(32),
            nothing,
            0,
            N_PSRN_INPUT,
            psrn,
            max_samples
        )
    end
end

function select_top_subtrees(common_subtrees::Dict{Node,Int}, n::Int, options::AbstractOptions)
    filtered_subtrees = filter(pair -> begin
        node = pair.first
        complexity = compute_complexity(node, options)
        return complexity <= 20
    end, common_subtrees)
    
    sorted_subtrees = sort(collect(filtered_subtrees), by=x->(x[2] * (1.0 + 0.3 * randn())), rev=true)
    
    result = Node[]
    
    for i in 1:min(n, length(sorted_subtrees))
        push!(result, sorted_subtrees[i][1])
    end
    
    while length(result) < n
        current_num = (length(result) - length(sorted_subtrees) + 1) ÷ 2 + 1
        is_positive = (length(result) - length(sorted_subtrees)) % 2 == 0
        val = is_positive ? Float32(current_num) : Float32(-current_num)
        push!(result, Node(; val=val))
    end
    
    return result
end

function evaluate_subtrees(subtrees::Vector{Node}, dataset::Dataset, options::AbstractOptions)
    n_samples = size(dataset.X, 2)  # Use the number of columns as the number of samples
    n_subtrees = length(subtrees)
    
    # Create a result matrix - using the same type as dataset.X
    T = eltype(dataset.X)
    result = zeros(T, n_samples, n_subtrees)
    
    # @info "n_subtrees: $n_subtrees"
    # @info "n_samples: $n_samples"

    # Evaluate each subtree
    for (i, subtree) in enumerate(subtrees)
        if isnothing(subtree)
            result[:, i] .= one(T)
        else
            # Creates an Expression object, providing the necessary parameters
            # @info "Evaluating subtree: $subtree"  # Print the Node object first
            
            # Use operators in options when creating an Expression
            expr = Expression(
                subtree,
                operators=options.operators,  # Use operators in options
                variable_names=dataset.variable_names  # Get variable_names from dataset
            )
            
            # Evaluate on data set X
            # @info "Starting eval_tree_array..."
            output, success = eval_tree_array(
                expr, 
                dataset.X  # Just use X, no transpose
            )
            # @info "eval_tree_array completed" success=success output_size=size(output)
            
            if success
                # If the output is one-dimensional, it is assigned directly to the corresponding column
                if length(output) == n_samples
                    result[:, i] = output
                    # @info "Successfully assigned output to result[:, $i]"
                else
                    @warn "Dimension mismatch: output length $(length(output)) doesn't match expected size ($n_samples). Using ones."
                    result[:, i] .= one(T)
                end
            else
                result[:, i] .= one(T)
                @warn "eval_tree_array failed for subtree $i, using ones"
            end
        end
    end
    
    # @info "Evaluation complete" result_size=size(result)
    return result
end

function analyze_common_subtrees(trees::Vector{<:Expression})
    # TODO - This is obviously not efficient, but it works for now

    subtree_counts = Dict{Node, Int}()
    
    for tree in trees
        if !isnothing(tree.tree)
            subtrees = get_subtrees(tree)
            for subtree in subtrees
                subtree_counts[subtree] = get(subtree_counts, subtree, 0) + 1
            end
        end
    end
    
    threshold = length(trees) * 0.01 # TODO need to adjust this threshold in tghe future
    common_patterns = filter(p -> p.second >= threshold, subtree_counts)
    
    if !isempty(common_patterns)
        # println("\nCommon subtree patterns:")
        for (pattern, count) in common_patterns
            # println("- $(string_tree(pattern)) (appeared $count times)")
            # @info pattern
        end
    end
    
    return common_patterns
end

# Gets all the subtrees of an expression tree
function get_subtrees(expr::Expression)
    if isnothing(expr.tree)
        return Node[]
    end
    return get_subtrees(expr.tree)
end

function get_subtrees(node::Node)
    subtrees = Node[]
    if isnothing(node)
        return subtrees
    end
    
    push!(subtrees, node)
    
    # Recursive processing of left and right subtrees
    if isdefined(node, :l) && !isnothing(node.l)
        append!(subtrees, get_subtrees(node.l))
    end
    
    if isdefined(node, :r) && !isnothing(node.r)
        append!(subtrees, get_subtrees(node.r))
    end
    
    return subtrees
end

get_subtrees(x::Number) = Node[]
get_subtrees(x::Symbol) = Node[]

function start_psrn_task(
    manager::PSRNManager,
    dominating_trees::Vector{<:Expression},
    dataset::Dataset,
    options::AbstractOptions,
    N_PSRN_INPUT::Int
)
    manager.call_count += 1

    # Check if PSRN needs to be executed
    # TODO - This is obviously not efficient, but it works for now
    if manager.call_count % 1 != 0 || (manager.current_task !== nothing && !istaskdone(manager.current_task))
        return
    end
    
    @info "Starting PSRN computation ($(manager.call_count ÷ 1)/1 times)"
    
    manager.current_task = Threads.@spawn begin # export JULIA_NUM_THREADS=4
        try

            common_subtrees = analyze_common_subtrees(dominating_trees)

            top_subtrees = select_top_subtrees(common_subtrees, N_PSRN_INPUT, options)

            # @info "Selected subtrees:" top_subtrees

            X_mapped = evaluate_subtrees(top_subtrees, dataset, options)
            
            # 添加降采样逻辑
            n_samples = size(X_mapped, 1)
            if n_samples > manager.max_samples
                # 随机选择max_samples个样本点
                sample_indices = randperm(n_samples)[1:manager.max_samples]
                X_mapped_sampled = X_mapped[sample_indices, :]
                
                # 检查 dataset.y 的维度
                y_dims = size(dataset.y)
                if length(y_dims) == 1
                    y_sampled = dataset.y[sample_indices]
                else
                    y_sampled = dataset.y[:, sample_indices]
                end
            else
                X_mapped_sampled = X_mapped
                y_sampled = dataset.y
            end


            # 添加调试信息
            # @info "Dimensions:" X_mapped_size=size(X_mapped_sampled) y_size=size(y_sampled)

            best_expressions, mse_values = get_best_expressions(
                manager.net, 
                X_mapped_sampled,
                y_sampled,
                top_subtrees, 
                options, 
                top_k=100
            )

            put!(manager.channel, best_expressions)

            # @info "best_expressions: $best_expressions"
        catch e
            bt = stacktrace(catch_backtrace())
            @error """
            PSRN task execution error:
            Error type: $(typeof(e))
            Error message: $e
            Error location: $(bt[1])
            Full stack:
            $(join(string.(bt), "\n"))
            """
        end
    end
end

# Check and process PSRN results
function process_psrn_results!(
    manager::PSRNManager,
    hall_of_fame::HallOfFame,
    dataset::Dataset,
    options::AbstractOptions
)
    while isready(manager.channel)
        new_expressions = take!(manager.channel)
        if !isempty(new_expressions)
            for psrn_expr in new_expressions
                # Create a new Expression using target type
                converted_expr = Expression(
                    psrn_expr.tree;  # Only keep the tree structure
                    operators=nothing,  # Set to nothing
                    variable_names=nothing  # Set to nothing
                )
                
                member = PopMember(dataset, converted_expr, options; deterministic=false)
                # @info "PSRN member: $member"
                # @info "type of member: $(typeof(member))"
                update_hall_of_fame!(hall_of_fame, [member], options)
            end
            @info "Added PSRN results to hall of fame"
        end
    end
end


function _main_search_loop!(
    state::AbstractSearchState{T,L,N},
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
) where {T,L,N}
    ropt.verbosity > 0 && @info "Started!"
    nout = length(datasets)
    start_time = time()
    progress_bar = if ropt.progress
        #TODO: need to iterate this on the max cycles remaining!
        sum_cycle_remaining = sum(state.cycles_remaining)
        WrappedProgressBar(
            sum_cycle_remaining, ropt.niterations; barlen=options.terminal_width
        )
    else
        nothing
    end

    last_print_time = time()
    last_speed_recording_time = time()
    num_evals_last = sum(sum, state.num_evals)
    num_evals_since_last = sum(sum, state.num_evals) - num_evals_last  # i.e., start at 0
    print_every_n_seconds = 5
    equation_speed = Float32[]

    N_PSRN_INPUT = 10
    psrn_manager = PSRNManager(
        N_PSRN_INPUT = N_PSRN_INPUT,
        operators = ["Add", "Mul", "Sub", "Div", "Identity", "Cos", "Sin", "Exp", "Log"],
        # operators = ["Add", "Mul", "Sub", "Div", "Identity"],
        n_symbol_layers = 2,
        options = options,
        max_samples = 100
        # max_samples = 10
    )

    if ropt.parallelism in (:multiprocessing, :multithreading)
        for j in 1:nout, i in 1:(options.populations)
            # Start listening for each population to finish:
            t = @async put!(state.channels[j][i], fetch(state.worker_output[j][i]))
            push!(state.tasks[j], t)
        end
    end
    kappa = 0
    resource_monitor = ResourceMonitor(;
        # Storing n times as many monitoring intervals as populations seems like it will
        # help get accurate resource estimates:
        max_recordings=options.populations * 100 * nout,
        start_reporting_at=options.populations * 3 * nout,
        window_size=options.populations * 2 * nout,
    )
    while sum(state.cycles_remaining) > 0
        kappa += 1
        if kappa > options.populations * nout
            kappa = 1
        end
        # nout, populations:
        j, i = state.task_order[kappa]

        # Check if error on population:
        if ropt.parallelism in (:multiprocessing, :multithreading)
            if istaskfailed(state.tasks[j][i])
                fetch(state.tasks[j][i])
                error("Task failed for population")
            end
        end
        # Non-blocking check if a population is ready:
        population_ready = if ropt.parallelism in (:multiprocessing, :multithreading)
            # TODO: Implement type assertions based on parallelism.
            isready(state.channels[j][i])
        else
            true
        end
        record_channel_state!(resource_monitor, population_ready)

        # Don't start more if this output has finished its cycles:
        # TODO - this might skip extra cycles?
        population_ready &= (state.cycles_remaining[j] > 0)
        if population_ready
            # Take the fetch operation from the channel since its ready
            (cur_pop, best_seen, cur_record, cur_num_evals) = if ropt.parallelism in
                (
                :multiprocessing, :multithreading
            )
                take!(
                    state.channels[j][i]
                )
            else
                state.worker_output[j][i]
            end::DefaultWorkerOutputType{Population{T,L,N},HallOfFame{T,L,N}}
            state.last_pops[j][i] = copy(cur_pop)
            state.best_sub_pops[j][i] = best_sub_pop(cur_pop; topn=options.topn)
            @recorder state.record[] = recursive_merge(state.record[], cur_record)
            state.num_evals[j][i] += cur_num_evals
            dataset = datasets[j]
            cur_maxsize = state.cur_maxsizes[j]

            for member in cur_pop.members
                size = compute_complexity(member, options)
                update_frequencies!(state.all_running_search_statistics[j]; size)
            end
            #! format: off
            update_hall_of_fame!(state.halls_of_fame[j], cur_pop.members, options)
            update_hall_of_fame!(state.halls_of_fame[j], best_seen.members[best_seen.exists], options)
            #! format: on

            dominating = calculate_pareto_frontier(state.halls_of_fame[j])
            
            dominating_trees = [member.tree for member in dominating]

            if true # if use PSRN
                start_psrn_task(psrn_manager, dominating_trees, dataset, options, N_PSRN_INPUT)
                process_psrn_results!(
                    psrn_manager,
                    state.halls_of_fame[j],
                    dataset,
                    options
                )
            end

            if options.save_to_file
                save_to_file(dominating, nout, j, dataset, options, ropt)
            end
            ###################################################################
            # Migration #######################################################
            if options.migration
                best_of_each = Population([
                    member for pop in state.best_sub_pops[j] for member in pop.members
                ])
                migrate!(
                    best_of_each.members => cur_pop, options; frac=options.fraction_replaced
                )
            end
            if options.hof_migration && length(dominating) > 0
                migrate!(dominating => cur_pop, options; frac=options.fraction_replaced_hof)
            end
            ###################################################################

            state.cycles_remaining[j] -= 1
            if state.cycles_remaining[j] == 0
                break
            end
            worker_idx = assign_next_worker!(
                state.worker_assignment;
                out=j,
                pop=i,
                parallelism=ropt.parallelism,
                state.procs,
            )
            iteration = if options.use_recorder
                key = "out$(j)_pop$(i)"
                find_iteration_from_record(key, state.record[]) + 1
            else
                0
            end

            c_rss = deepcopy(state.all_running_search_statistics[j])
            in_pop = copy(cur_pop::Population{T,L,N})
            state.worker_output[j][i] = @sr_spawner(
                begin
                    _dispatch_s_r_cycle(
                        in_pop,
                        dataset,
                        options;
                        pop=i,
                        out=j,
                        iteration,
                        ropt.verbosity,
                        cur_maxsize,
                        running_search_statistics=c_rss,
                    )
                end,
                parallelism = ropt.parallelism,
                worker_idx = worker_idx
            )
            if ropt.parallelism in (:multiprocessing, :multithreading)
                state.tasks[j][i] = @async put!(
                    state.channels[j][i], fetch(state.worker_output[j][i])
                )
            end

            total_cycles = ropt.niterations * options.populations
            state.cur_maxsizes[j] = get_cur_maxsize(;
                options, total_cycles, cycles_remaining=state.cycles_remaining[j]
            )
            move_window!(state.all_running_search_statistics[j])
            if !isnothing(progress_bar)
                head_node_occupation = estimate_work_fraction(resource_monitor)
                update_progress_bar!(
                    progress_bar,
                    only(state.halls_of_fame),
                    only(datasets),
                    options,
                    equation_speed,
                    head_node_occupation,
                    ropt.parallelism,
                )
            end
            if ropt.logger !== nothing
                logging_callback!(ropt.logger; state, datasets, ropt, options)
            end
        end
        yield()

        ################################################################
        ## Search statistics
        elapsed_since_speed_recording = time() - last_speed_recording_time
        if elapsed_since_speed_recording > 1.0
            num_evals_since_last, num_evals_last = let s = sum(sum, state.num_evals)
                s - num_evals_last, s
            end
            current_speed = num_evals_since_last / elapsed_since_speed_recording
            push!(equation_speed, current_speed)
            average_over_m_measurements = 20 # 20 second running average
            if length(equation_speed) > average_over_m_measurements
                deleteat!(equation_speed, 1)
            end
            last_speed_recording_time = time()
        end
        ################################################################

        ################################################################
        ## Printing code
        elapsed = time() - last_print_time
        # Update if time has passed
        if elapsed > print_every_n_seconds
            if ropt.verbosity > 0 && !ropt.progress && length(equation_speed) > 0

                # Dominating pareto curve - must be better than all simpler equations
                head_node_occupation = estimate_work_fraction(resource_monitor)
                total_cycles = ropt.niterations * options.populations
                print_search_state(
                    state.halls_of_fame,
                    datasets;
                    options,
                    equation_speed,
                    total_cycles,
                    state.cycles_remaining,
                    head_node_occupation,
                    parallelism=ropt.parallelism,
                    width=options.terminal_width,
                )
            end
            last_print_time = time()
        end
        ################################################################

        ################################################################
        ## Early stopping code
        if any((
            check_for_loss_threshold(state.halls_of_fame, options),
            check_for_user_quit(state.stdin_reader),
            check_for_timeout(start_time, options),
            check_max_evals(state.num_evals, options),
        ))
            break
        end
        ################################################################
    end
    if !isnothing(progress_bar)
        finish!(progress_bar)
    end
    return nothing
end


function _tear_down!(
    state::AbstractSearchState, ropt::AbstractRuntimeOptions, options::AbstractOptions
)
    close_reader!(state.stdin_reader)
    # Safely close all processes or threads
    if ropt.parallelism == :multiprocessing
        state.we_created_procs && rmprocs(state.procs)
    elseif ropt.parallelism == :multithreading
        nout = length(state.worker_output)
        for j in 1:nout, i in eachindex(state.worker_output[j])
            wait(state.worker_output[j][i])
        end
    end
    @recorder json3_write(state.record[], options.recorder_file)
    return nothing
end
function _format_output(
    state::AbstractSearchState,
    datasets,
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
)
    nout = length(datasets)
    out_hof = if ropt.dim_out == 1
        embed_metadata(only(state.halls_of_fame), options, only(datasets))
    else
        map(Fix{2}(embed_metadata, options), state.halls_of_fame, datasets)
    end
    if ropt.return_state
        return (map(Fix{2}(embed_metadata, options), state.last_pops, datasets), out_hof)
    else
        return out_hof
    end
end

@stable default_mode = "disable" function _dispatch_s_r_cycle(
    in_pop::Population{T,L,N},
    dataset::Dataset,
    options::AbstractOptions;
    pop::Int,
    out::Int,
    iteration::Int,
    verbosity,
    cur_maxsize::Int,
    running_search_statistics,
) where {T,L,N}
    record = RecordType()
    @recorder record["out$(out)_pop$(pop)"] = RecordType(
        "iteration$(iteration)" => record_population(in_pop, options)
    )
    num_evals = 0.0
    normalize_frequencies!(running_search_statistics)
    out_pop, best_seen, evals_from_cycle = s_r_cycle(
        dataset,
        in_pop,
        options.ncycles_per_iteration,
        cur_maxsize,
        running_search_statistics;
        verbosity=verbosity,
        options=options,
        record=record,
    )
    num_evals += evals_from_cycle
    out_pop, evals_from_optimize = optimize_and_simplify_population(
        dataset, out_pop, options, cur_maxsize, record
    )
    num_evals += evals_from_optimize
    if options.batching
        for i_member in 1:(options.maxsize)
            score, result_loss = score_func(dataset, best_seen.members[i_member], options)
            best_seen.members[i_member].score = score
            best_seen.members[i_member].loss = result_loss
            num_evals += 1
        end
    end
    return (out_pop, best_seen, record, num_evals)
end
function _info_dump(
    state::AbstractSearchState,
    datasets::Vector{D},
    ropt::AbstractRuntimeOptions,
    options::AbstractOptions,
) where {D<:Dataset}
    ropt.verbosity <= 0 && return nothing

    nout = length(state.halls_of_fame)
    if nout > 1
        @info "Final populations:"
    else
        @info "Final population:"
    end
    for (j, (hall_of_fame, dataset)) in enumerate(zip(state.halls_of_fame, datasets))
        if nout > 1
            @info "Output $j:"
        end
        equation_strings = string_dominating_pareto_curve(
            hall_of_fame,
            dataset,
            options;
            width=@something(
                options.terminal_width,
                ropt.progress ? displaysize(stdout)[2] : nothing,
                Some(nothing)
            )
        
        )
        println(equation_strings)
    end

    if options.save_to_file
        output_directory = joinpath(
            something(options.output_directory, "outputs"), ropt.run_id
        )
        @info "Results saved to:"
        for j in 1:nout
            filename = nout > 1 ? "hall_of_fame_output$(j).csv" : "hall_of_fame.csv"
            output_file = joinpath(output_directory, filename)
            println("  - ", output_file)
        end
    end
    return nothing
end

include("MLJInterface.jl")
using .MLJInterfaceModule: SRRegressor, MultitargetSRRegressor

# Hack to get static analysis to work from within tests:
@ignore include("../test/runtests.jl")

# TODO: Hack to force ConstructionBase version
using ConstructionBase: ConstructionBase as _

include("precompile.jl")
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        do_precompilation(Val(:precompile))
    end
end

end #module SR
