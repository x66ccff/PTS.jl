module PSRNmodel

using ..PSRNfunctions
import ..CoreModule.OperatorsModule:
    plus,
    sub,
    mult,
    square,
    cube,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma

import ..CoreModule: Options, Dataset

using KernelAbstractions
const KA = KernelAbstractions
using CUDA

@static if Base.find_package("AMDGPU") !== nothing
    using AMDGPU
    using AMDGPU: ROCArray
end

@static if Base.find_package("oneAPI") !== nothing
    using oneAPI
    using oneAPI: oneArray
end

using Printf: @sprintf
using DynamicExpressions: Node, Expression

abstract type Operator end

struct UnaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    op::Function
end

struct BinaryOperator <: Operator
    name::String
    kernel::Function
    is_directed::Bool
    op::Function
end

const OPERATORS = Dict{String,Operator}(
    "Identity" => UnaryOperator("Identity", identity_kernel!, true, identity),
    "Sin" => UnaryOperator("Sin", sin_kernel!, true, sin),
    "Cos" => UnaryOperator("Cos", cos_kernel!, true, cos),
    "Exp" => UnaryOperator("Exp", exp_kernel!, true, exp),
    "Log" => UnaryOperator("Log", log_kernel!, true, safe_log),
    "Neg" => UnaryOperator("Neg", neg_kernel!, true, -),
    "Add" => BinaryOperator("Add", add_kernel!, false, +),
    "Mul" => BinaryOperator("Mul", mul_kernel!, false, *),
    "Div" => BinaryOperator("Div", div_kernel!, true, /),
    "Sub" => BinaryOperator("Sub", sub_kernel!, true, -),
    "Pow" => BinaryOperator("Pow", pow_kernel!, true, safe_pow),
    "Sqrt" => UnaryOperator("Sqrt", sqrt_kernel!, true, safe_sqrt),
)

# SymbolLayer
mutable struct SymbolLayer
    in_dim::Int
    out_dim::Int
    operators::Vector{Operator}  # use Operator in SR.jl instead of String
    n_binary_U::Int  # undirected (+ *)
    n_binary_D::Int  # directed (/ -)
    n_unary::Int
    operator_list::Vector{Operator}
    n_triu::Int
    in_dim_square::Int
    out_dim_cum_ls::Union{Vector{Int},Nothing}
    offset_tensor::Union{Matrix{Int},Nothing}

    function SymbolLayer(in_dim::Int, operator_names::Vector{String})
        n_binary_U = 0
        n_binary_D = 0
        n_unary = 0
        operator_list = Operator[]
        operators = [OPERATORS[name] for name in operator_names]

        n_triu = (in_dim * (in_dim + 1)) ÷ 2
        in_dim_square = in_dim * in_dim

        # count the numbers of operators
        for op in operators
            if op isa BinaryOperator
                if op.is_directed
                    n_binary_D += 1
                else
                    n_binary_U += 1
                end
            else
                n_unary += 1
            end
        end

        # Add operators in order: first undirected binary, then directed binary, and finally unary
        # 1. Undirected binary operators
        for op in operators
            if op isa BinaryOperator && !op.is_directed
                push!(operator_list, op)
            end
        end

        # 2. Directed binary operators
        for op in operators
            if op isa BinaryOperator && op.is_directed
                push!(operator_list, op)
            end
        end

        # 3. Unary operators
        for op in operators
            if op isa UnaryOperator
                push!(operator_list, op)
            end
        end

        out_dim = n_unary * in_dim + n_binary_U * n_triu + n_binary_D * in_dim_square

        return new(
            in_dim,
            out_dim,
            operators,
            n_binary_U,
            n_binary_D,
            n_unary,
            operator_list,
            n_triu,
            in_dim_square,
            nothing,
            nothing,
        )
    end
end

# Modify Index Generation Function to Use Passed-in Backend
function get_triu_indices(n::Int, backend)
    if backend isa KA.CPU
        indices = Tuple{Int,Int}[]
        for i in 1:n
            for j in i:n
                push!(indices, (i, j))
            end
        end
        return indices
    else
        # return two vectors in GPU version
        row_idx = Int[]
        col_idx = Int[]
        for i in 1:n
            for j in i:n
                push!(row_idx, i)
                push!(col_idx, j)
            end
        end
        # Returns the corresponding array based on the backend type
        return (to_device(row_idx, backend), to_device(col_idx, backend))
    end
end

function to_device(x::AbstractArray, backend)
    if isa(backend, KA.GPU) && CUDA.functional()
        return CuArray(x)
    elseif @isdefined(ROCArray) && isa(backend, KA.GPU) && AMDGPU.functional()
        return ROCArray(x)
    elseif @isdefined(oneArray) && isa(backend, KA.GPU) && oneAPI.functional()
        return oneArray(x)
    end
    return x
end

function get_op_and_offset(layer::SymbolLayer, index::Int)
    out_dim_cum_ls = get_out_dim_cum_ls(layer)

    # Find the corresponding operator
    op_idx = 1
    for i in eachindex(out_dim_cum_ls)
        if index < out_dim_cum_ls[i]
            op_idx = i
            break
        end
    end

    # Get offset
    offset = layer.offset_tensor[index, :]
    return layer.operator_list[op_idx], offset
end

# Add a forward propagator
function forward(layer::SymbolLayer, x::AbstractArray, backend)
    results = []

    for op in layer.operator_list
        if op isa UnaryOperator
            # Back end to get input data
            device_backend = get_backend(x)
            # Create and execute the kernel
            kernel = op.kernel(device_backend, 256)
            result = similar(x)
            event = kernel(result, x; ndrange=size(x))
            if event !== nothing
                wait(event)
            end
            push!(results, result)
        else # BinaryOperator
            if op.is_directed
                # Directed binary operation (division, subtraction)
                device_backend = get_backend(x)
                kernel = op.kernel(device_backend, 256)

                if device_backend isa KA.CPU
                    for i in 1:(layer.in_dim)
                        for j in 1:(layer.in_dim)
                            x1 = view(x, :, i)
                            x2 = view(x, :, j)
                            result = similar(x1)
                            event = kernel(result, x1, x2; ndrange=size(x1))
                            if event !== nothing
                                wait(event)
                            end
                            push!(results, result)
                        end
                    end
                else
                    # GPU version: Batch processing
                    row_idx, col_idx = get_square_indices(layer.in_dim, device_backend)
                    x1 = view(x, :, row_idx)
                    x2 = view(x, :, col_idx)
                    result = similar(x1)
                    event = kernel(result, x1, x2; ndrange=size(x1))
                    if event !== nothing
                        wait(event)
                    end
                    push!(results, result)
                end
            else
                # Undirected binary operation (e.g. addition, multiplication)
                device_backend = get_backend(x)
                kernel = op.kernel(device_backend, 256)

                if device_backend isa KA.CPU
                    for i in 1:(layer.in_dim)
                        for j in i:(layer.in_dim)
                            x1 = view(x, :, i)
                            x2 = view(x, :, j)
                            result = similar(x1)
                            event = kernel(result, x1, x2; ndrange=size(x1))
                            if event !== nothing
                                wait(event)
                            end
                            push!(results, result)
                        end
                    end
                else
                    # GPU version: Batch processing
                    row_idx, col_idx = get_triu_indices(layer.in_dim, device_backend)
                    x1 = view(x, :, row_idx)
                    x2 = view(x, :, col_idx)
                    result = similar(x1)
                    event = kernel(result, x1, x2; ndrange=size(x1))
                    if event !== nothing
                        wait(event)
                    end
                    push!(results, result)
                end
            end
        end
    end

    return hcat(results...)
end

function get_backend(x::AbstractArray)
    if x isa CuArray
        return CUDA.CUDABackend()
    elseif x isa ROCArray
        return AMDGPU.ROCBackend()
    elseif x isa oneArray
        return oneAPI.oneBackend()
    else
        return KA.CPU()
    end
end

mutable struct PSRN
    n_variables::Int
    operators::Vector{Operator}
    n_symbol_layers::Int
    layers::Vector{SymbolLayer}
    current_exprs::Vector{Expression}
    out_dim::Int
    backend::Any
    options::Options

    function PSRN(;
        n_variables::Int=1,
        operators::Vector{String}=["Add", "Mul", "Identity", "Sin", "Exp", "Neg", "Inv"],
        n_symbol_layers::Int=2,
        backend=KA.CPU(),
        initial_expressions=nothing,
        options=nothing,
    )

        # options = Options(;
        #     binary_operators=[+, -, *, /, ^],
        #     unary_operators=[cos, exp, sin, log],  # kkTODO
        #     populations=20,
        #     parsimony=0.0001 
        # )

        layers = SymbolLayer[]
        for i in 1:n_symbol_layers
            in_dim = i == 1 ? n_variables : layers[end].out_dim
            layer = SymbolLayer(in_dim, operators)
            init_offset(layer, backend)
            push!(layers, layer)
        end

        # Create initial expression
        variable_names = ["x$i" for i in 1:n_variables]

        # Process based on the type of initial_expressions
        current_exprs = if isnothing(initial_expressions)
            # Variable expressions are used by default
            [
                Expression(
                    Node(Float32; feature=i);
                    operators=options.operators,
                    variable_names=variable_names,
                ) for i in 1:n_variables
            ]
        elseif initial_expressions isa Vector{Node}
            # If it is a Node array, convert it to an Expression array
            [
                Expression(
                    node; operators=options.operators, variable_names=variable_names
                ) for node in initial_expressions
            ]
        elseif initial_expressions isa Vector{Expression}
            # If it is already an Expression array, use it directly
            initial_expressions
        else
            throw(
                ArgumentError(
                    "initial_expressions must be Nothing, Vector{Node}, or Vector{Expression}",
                ),
            )
        end

        operator_list = [OPERATORS[name] for name in operators]
        out_dim = layers[end].out_dim

        return new(
            n_variables,
            operator_list,
            n_symbol_layers,
            layers,
            current_exprs,
            out_dim,
            backend,
            options,
        )
    end
end

function _get_expr(psrn::PSRN, index::Int, layer_idx::Int)
    if layer_idx < 1
        return psrn.current_exprs[index]
    end

    layer = psrn.layers[layer_idx]
    op, offsets = get_op_and_offset(layer, index)

    # Get subexpression
    expr1 = _get_expr(psrn, offsets[1], layer_idx - 1)
    T = eltype(expr1.tree)  # Gets the type of the expression

    if op isa UnaryOperator
        # Create a unary operation expression
        if op.name == "Identity"
            return expr1
        end
        return op.op(expr1)
    else
        # Create a binary operation expression
        expr2 = _get_expr(psrn, offsets[2], layer_idx - 1)
        return op.op(expr1, expr2)
    end
end

function get_expr(psrn::PSRN, index::Int)
    return _get_expr(psrn, index, length(psrn.layers))
end

# Add a forward propagation function for PSRN
function forward(psrn::PSRN, x::AbstractArray{T}) where {T}
    # Check input dimension
    size(x, 2) == psrn.n_variables || throw(
        DimensionMismatch(
            "Input should have $(psrn.n_variables) features, got $(size(x, 2))"
        ),
    )

    # Make sure the data is on the correct device
    x_device = to_device(x, psrn.backend)

    # Forward propagation
    h = x_device
    for layer in psrn.layers
        h = forward(layer, h, psrn.backend)
    end
    return h
end

function get_out_dim_cum_ls(layer::SymbolLayer)
    if layer.out_dim_cum_ls !== nothing
        return layer.out_dim_cum_ls
    end

    out_dim_ls = Int[]
    for func in layer.operator_list
        if func isa UnaryOperator
            push!(out_dim_ls, layer.in_dim)
        else
            if func isa BinaryOperator
                if func.is_directed
                    push!(out_dim_ls, layer.in_dim_square)
                else
                    push!(out_dim_ls, layer.n_triu)
                end
            end
        end
    end

    layer.out_dim_cum_ls = [sum(out_dim_ls[1:i]) for i in 1:length(out_dim_ls)]
    return layer.out_dim_cum_ls
end

function init_offset(layer::SymbolLayer, backend)
    return layer.offset_tensor = get_offset_tensor(layer, backend)
end

function get_offset_tensor(layer::SymbolLayer, backend)
    offset_tensor = zeros(Int, layer.out_dim, 2)
    arange_tensor = collect(1:(layer.in_dim))

    binary_U_tensor = zeros(Int, layer.n_triu, 2)
    binary_D_tensor = zeros(Int, layer.in_dim_square, 2)
    unary_tensor = zeros(Int, layer.in_dim, 2)

    unary_tensor[:, 1] = arange_tensor
    unary_tensor[:, 2] .= layer.in_dim

    # Fill binary_U_tensor(index of undirected binary operation)
    start = 1
    for i in 1:(layer.in_dim)
        len = layer.in_dim - i + 1
        binary_U_tensor[start:(start + len - 1), 1] .= i
        binary_U_tensor[start:(start + len - 1), 2] = i:(layer.in_dim)
        start += len
    end

    # Fill binary_D_tensor(index of directed binary operation)
    start = 1
    for i in 1:(layer.in_dim)
        len = layer.in_dim
        binary_D_tensor[start:(start + len - 1), 1] .= i
        binary_D_tensor[start:(start + len - 1), 2] = 1:(layer.in_dim)
        start += len
    end

    # Combine all indexes
    start = 1
    for func in layer.operator_list
        if func isa UnaryOperator
            t = unary_tensor
        else
            t = func.is_directed ? binary_D_tensor : binary_U_tensor
        end
        len = size(t, 1)
        offset_tensor[start:(start + len - 1), :] = t
        start += len
    end

    return offset_tensor
end

# Add print method
function Base.show(io::IO, psrn::PSRN)
    print(
        io,
        "PSRN(n_variables=$(psrn.n_variables), operators=$(psrn.operators), " *
        "n_layers=$(psrn.n_symbol_layers))\n",
    )
    print(io, "Layer dimensions: ")
    return print(io, join([layer.out_dim for layer in psrn.layers], " → "))
end

function to_device(psrn::PSRN, backend)
    # Create a new PSRN instance and update backend
    new_psrn = PSRN(;
        n_variables=psrn.n_variables,
        operators=[op.name for op in psrn.operators],
        n_symbol_layers=psrn.n_symbol_layers,
        backend=backend,
        initial_expressions=psrn.current_exprs,  # Pass the current base expression from hall of fame
    )
    return new_psrn
end

function get_square_indices(n::Int, backend)
    if backend isa KA.CPU
        indices = Tuple{Int,Int}[]
        for i in 1:n
            for j in 1:n
                push!(indices, (i, j))
            end
        end
        return indices
    else
        # The GPU version returns two vectors
        row_idx = Int[]
        col_idx = Int[]
        for i in 1:n
            for j in 1:n
                push!(row_idx, i)
                push!(col_idx, j)
            end
        end
        return (to_device(row_idx, backend), to_device(col_idx, backend))
    end
end

function get_preferred_backend()
    if CUDA.functional()
        return CUDA.CUDABackend()
    elseif @isdefined(AMDGPU) && AMDGPU.functional()
        return AMDGPU.ROCBackend()
    elseif @isdefined(oneAPI) && oneAPI.functional()
        return oneAPI.oneBackend()
    end
    return KernelAbstractions.CPU()
end

function to_device(x::AbstractArray, backend::Union{Module,KA.Backend})
    if backend isa KA.GPU
        if CUDA.functional()
            return CuArray(x)
        elseif @isdefined(AMDGPU) && AMDGPU.functional()
            return ROCArray(x)
        elseif @isdefined(oneAPI) && oneAPI.functional()
            return oneArray(x)
        end
    end
    return Array(x)
end

function find_best_indices(outputs::AbstractArray, y::AbstractArray; top_k::Int=100)
    backend = outputs isa CUDA.CuArray ? CUDA : CPU
    y_device = to_device(y, backend)

    # Calculate the MSE for each output with respect to the target value
    n_samples = size(outputs, 1)
    n_expressions = size(outputs, 2)

    # Initialize the error accumulator
    sum_squared_errors = CUDA.zeros(eltype(outputs), n_expressions)

    # Calculate the MSE for each expression
    for i in 1:n_samples
        diff = outputs[i, :] .- y_device[i]
        sum_squared_errors .+= diff .^ 2
    end
    mean_squared_errors = sum_squared_errors ./ n_samples
    # @info "Mean squared errors before handling NaN/Inf" mean_squared_errors

    # Move the data back to the CPU for processing
    mean_squared_errors_cpu = Array(mean_squared_errors)

    # Handle invalid values on the CPU
    mean_squared_errors_cpu[isnan.(mean_squared_errors_cpu)] .= Inf32
    mean_squared_errors_cpu[isinf.(mean_squared_errors_cpu)] .= Inf32

    # @info "Mean squared errors after handling NaN/Inf" mean_squared_errors_cpu

    # Find the indices of the top_k smallest MSEs
    sorted_indices = partialsortperm(
        mean_squared_errors_cpu, 1:min(top_k, length(mean_squared_errors_cpu))
    )

    # Return the indices and corresponding MSE values
    return sorted_indices, mean_squared_errors_cpu[sorted_indices]
end

function get_best_expressions(
    psrn::PSRN, X::AbstractArray, y::AbstractArray; top_k::Int=100
)
    backend = get_preferred_backend()
    X_device = to_device(X, backend)

    outputs = forward(psrn, X_device)

    best_indices, mse_values = find_best_indices(outputs, y; top_k=top_k)

    best_expressions = [get_expr(psrn, idx) for idx in best_indices]

    # println("Best expressions:")
    # println("-"^20)
    for (expr, mse) in zip(best_expressions, mse_values)
        # println("MSE: ", mse, " | Expression: ", expr)
    end
    # println("-"^20)

    return best_expressions, mse_values
end

export PSRN, forward, get_expr, to_device, find_best_indices, get_best_expressions

end
