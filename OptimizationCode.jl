import Pkg
Pkg.add("Plots")
using Random
using LinearAlgebra
using Statistics
using Plots
using DelimitedFiles

function elu(X)
    elu=exp.(X).-1
    return elu, X
end

function init_param(layer_dimensions, activation_functions)
    param = Dict()
    for l=1:length(layer_dimensions)-1
        param[string("W_" , string(l))] = 0.01f0*randn(layer_dimensions[l+1] , layer_dimensions[l])
        param[string("b_" , string(l))] = zeros(layer_dimensions[l+1] , 1)
		param[string("g_" , string(l))] = activation_functions[l]
    end
    return param
end

function update_param(parameters, grads, lambda, alpha, q, t0, tw, tb, method)
	L = Int(length(parameters)/3)
	for l = 0:(L-1)
        W=parameters[string("W_", string(l+1))]
        b=parameters[string("b_", string(l+1))]
        if method == 1
            parameters[string("W_", string(l+1))] = W - grads[string("dW_", string(l+1))]./lambda
            parameters[string("b_", string(l+1))] = b - grads[string("db_", string(l+1))]./lambda
        elseif method == 2
            if t0
                tw=W
                tb=b
                t0=false
            end
            alphaold=alpha
            radical=sqrt((alphaold^2-q)^2+4alphaold^2)
            addition=q-alphaold^2
            alpha=1/2*(addition-radical)
            if (alpha>1)|(alpha<0)
                alpha=1/2*(addition+radical)
            end
            beta=alphaold*(1-alphaold)/(alphaold^2+alpha)
            parameters[string("W_", string(l+1))] = tw - grads[string("dW_", string(l+1))]./lambda
            parameters[string("b_", string(l+1))] = tb - grads[string("db_", string(l+1))]./lambda
            tw=parameters[string("W_", string(l+1))] + beta*(parameters[string("W_", string(l+1))]-W)
            tb=parameters[string("b_", string(l+1))] + beta*(parameters[string("b_", string(l+1))]-b)
        end
	end
	return parameters, tw, tb, alpha
end

function forward_linear(A, w, b)
    Z = w*A .+ b
    cache = (A, w, b)
    return Z, cache
end

function calculate_activation_forward(A_pre, W, b, function_type)
    Z, linear_step_cache = forward_linear(A_pre, W, b)
    A, activation_step_cache = elu(Z)
    cache = (linear_step_cache, activation_step_cache, function_type)
    return A, cache
end

function model_forward_step(X, params)
    all_caches = []
    A = X
    L = length(params)/3
    for l = 1:L-1
        A_pre = A
        A, cache = calculate_activation_forward(A_pre,  params[string("W_", string(Int(l)))],
                                                        params[string("b_", string(Int(l)))],
                                                        params[string("g_", string(Int(l)))])
        push!(all_caches, cache)
    end
	A_l, cache = calculate_activation_forward(A, params[string("W_", string(Int(L)))],
												 params[string("b_", string(Int(L)))],
												 params[string("g_", string(Int(L)))])
 	push!(all_caches, cache)
    return A_l, all_caches
end

function cost_function(AL, Y)
    cost = mean(.5*(AL.-Y).^2)
    return cost
end

function backward_linear_step(dZ, cache)
    A_prev, W, b = cache
    m = size(A_prev)[2]
    dW = dZ * (A_prev')/m
    db = sum(dZ, dims = 2)/m
    dA_prev = (W')* dZ
    return dW, db, dA_prev
end

function backward_elu(dA, cache_activation)
    return dA.*(elu(cache_activation)[1].+1)

end

function backward_activation_step(dA, cache)
    linear_cache , cache_activation, activation = cache
    dZ = backward_elu(dA, cache_activation)
    dW, db, dA_prev = backward_linear_step(dZ, linear_cache)
    return dW, db, dA_prev
end

function (model_backwards_step(A_l, Y, caches))
    grads = Dict()
    L = length(caches)
    m = size(A_l)[2]
    Y = reshape(Y, size(A_l))
    dA_l = (A_l.-Y)
    current_cache = caches[L]
    grads[string("dW_", string(L))], grads[string("db_", string(L))], grads[string("dA_", string(L-1))] = backward_activation_step(dA_l, current_cache)
    for l = reverse(0:L-2)
        current_cache = caches[l+1]
        grads[string("dW_", string(l+1))], grads[string("db_", string(l+1))], grads[string("dA_", string(l))] = backward_activation_step(grads[string("dA_", string(l+1))], current_cache)
    end
    return grads
end


function train_nn(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, method)
    params = init_param(layers_dimensions, activation_functions)
    costs = []
    iters = []
	m = size(X,2)
    finalcost=0
    alpha=.75
    q=mu/lambda
    t0=true
    tw=1
    tb=1
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
        cost = cost_function(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params, tw, tb, alpha = update_param(params , grads , lambda, alpha, q, t0, tw, tb, method)
        push!(iters , i)
        push!(costs , cost)
        if i==n_iter
            finalcost=cost
        end
    end
    return costs, iters, finalcost
end

function graph_nn(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter)
    costs_grad, iters_grad, finalcost_grad=train_nn(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 1)
    costs_nest, iters_nest, finalcost_nest=train_nn(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 2)
    println("Gradient Descent Final Cost: ", finalcost_grad)
    println("Gradient Descent Time: ", @elapsed train_nn_nograph(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 1))
    println("Nesterov Method Final Cost: ", finalcost_nest)
    println("Nesterov Method Time: ", @elapsed train_nn_nograph(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 2))
    plt = plot(iters_grad, costs_grad ,title =  "Cost Function vs Number of Iterations", lab="Gradient Descent")
    plt = plot!(iters_nest, costs_nest, lab="Nesterov Method")
    xaxis!("Iterations")
    yaxis!("Cost")
    return plt
end

function train_nn_nograph(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, method)
    params = init_param(layers_dimensions, activation_functions)
	m = size(X,2)
    finalcost=0
    alpha=.75
    q=mu/lambda
    t0=true
    tw=1
    tb=1
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
        cost = cost_function(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params, tw, tb, alpha = update_param(params , grads , lambda, alpha, q, t0, tw, tb, method)
        if i==n_iter
            finalcost=cost
        end
    end
    return finalcost
end

function display_graph()
    layers_dimensions = (2,1)
    activation_functions = ["elu"]
    data=400
    X=rand(2,data).>.5
    Y=X[1,:].&X[2,:]
    Y=Y'
    lambda = 12
    mu=.1
    n_iter = 10
    return graph_nn(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter)
end

function cost_table(trials)
    layers_dimensions = (2,1)
    activation_functions = ["elu"]
    lambda = 6.5
    mu=.75
    cost_grad=0
    cost_nest=0
    time_grad=0
    time_nest=0
    data=40
    for n_iter in [10,20,40]
        println(n_iter)
        for i=1:trials
            X=rand(2,data).>.5
            Y=X[1,:].&X[2,:]
            Y=Y'
            time_grad=time_grad+@elapsed (newcost=train_nn_nograph(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 1))
            cost_grad=cost_grad+newcost
            time_nest=time_nest+@elapsed (newcost=train_nn_nograph(layers_dimensions, activation_functions, X, Y, lambda, mu, n_iter, 2))
            cost_nest=cost_nest+newcost
        end
        cost_grad=cost_grad/trials
        cost_nest=cost_nest/trials
        time_grad=time_grad/trials
        time_nest=time_nest/trials
        println("Gradient Descent Cost: ", cost_grad)
        println("Nesterov Method Cost:  ", cost_nest)
        println("Gradient Descent Time: ", time_grad)
        println("Nesterov Method Time:  ", time_nest)
    end
end


cost_table(100)
display_graph()
